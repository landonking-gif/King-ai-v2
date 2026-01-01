"""
LangSmith Integration - LLM tracing and evaluation.

Provides observability for all LLM interactions in King AI v2.
Enables tracing, evaluation, and debugging of LLM calls.

Features:
- Automatic tracing of LLM calls
- Performance metrics collection
- Error tracking and debugging
- Evaluation dataset management
"""

import uuid
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import dataclass, field

from config.settings import settings
from src.utils.logging import get_logger

logger = get_logger("langsmith")

# Check if LangSmith is available
LANGSMITH_AVAILABLE = False
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
    logger.info("LangSmith SDK available")
except ImportError:
    logger.info("LangSmith SDK not installed - tracing disabled")


@dataclass
class TraceMetadata:
    """Metadata for a trace."""
    run_id: str
    name: str
    run_type: str
    start_time: datetime
    inputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0


class LangSmithTracer:
    """
    LangSmith tracing integration for King AI.
    
    Provides comprehensive tracing for all LLM interactions,
    enabling debugging, evaluation, and performance monitoring.
    """
    
    def __init__(
        self,
        project_name: str = "king-ai-v2",
        api_key: Optional[str] = None
    ):
        """
        Initialize the LangSmith tracer.
        
        Args:
            project_name: LangSmith project name
            api_key: LangSmith API key (or from settings)
        """
        self.project_name = project_name or settings.langchain_project
        self.api_key = api_key or settings.langchain_api_key
        self.enabled = LANGSMITH_AVAILABLE and bool(self.api_key) and settings.langchain_tracing_v2
        self._client: Optional[Any] = None
        
        # Local trace storage for when LangSmith is disabled
        self._local_traces: Dict[str, TraceMetadata] = {}
        self._max_local_traces = 1000
        
        if self.enabled:
            try:
                self._client = Client(api_key=self.api_key)
                logger.info(
                    f"LangSmith tracing enabled",
                    project=self.project_name
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False
        else:
            logger.info("LangSmith tracing disabled (no API key or SDK)")
    
    @asynccontextmanager
    async def trace_llm_call(
        self,
        name: str,
        inputs: Dict[str, Any],
        run_type: str = "llm",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing LLM calls.
        
        Usage:
            async with tracer.trace_llm_call("my_call", {"prompt": "..."}) as trace:
                result = await llm.complete(...)
                trace["output"] = result
        
        Args:
            name: Name of the trace (e.g., "intent_classification")
            inputs: Input data to the LLM
            run_type: Type of run (llm, chain, tool, etc.)
            metadata: Additional metadata
            
        Yields:
            Dictionary to store output and error
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        trace_data = {"output": None, "error": None}
        
        # Create trace metadata
        trace_meta = TraceMetadata(
            run_id=run_id,
            name=name,
            run_type=run_type,
            start_time=start_time,
            inputs=self._truncate_dict(inputs),
            metadata=metadata or {}
        )
        
        try:
            # Create run in LangSmith if enabled
            if self.enabled and self._client:
                try:
                    self._client.create_run(
                        name=name,
                        run_type=run_type,
                        inputs=self._truncate_dict(inputs),
                        run_id=run_id,
                        project_name=self.project_name,
                        extra=metadata or {}
                    )
                except Exception as e:
                    logger.debug(f"Failed to create LangSmith run: {e}")
            
            yield trace_data
            
            # Record success
            trace_meta.end_time = datetime.utcnow()
            trace_meta.outputs = {"response": self._truncate_str(str(trace_data.get("output", "")))}
            
            if self.enabled and self._client:
                try:
                    self._client.update_run(
                        run_id=run_id,
                        outputs=trace_meta.outputs,
                        end_time=trace_meta.end_time
                    )
                except Exception as e:
                    logger.debug(f"Failed to update LangSmith run: {e}")
            
            logger.debug(
                f"LLM trace completed: {name}",
                run_id=run_id,
                duration_ms=trace_meta.duration_ms
            )
            
        except Exception as e:
            # Record error
            trace_meta.end_time = datetime.utcnow()
            trace_meta.error = str(e)
            trace_data["error"] = str(e)
            
            if self.enabled and self._client:
                try:
                    self._client.update_run(
                        run_id=run_id,
                        error=str(e),
                        end_time=trace_meta.end_time
                    )
                except Exception as update_err:
                    logger.debug(f"Failed to update LangSmith error: {update_err}")
            
            logger.error(f"LLM trace error: {name}", error=str(e))
            raise
        
        finally:
            # Store locally
            self._store_local_trace(trace_meta)
    
    def trace(self, name: str = None, run_type: str = "chain"):
        """
        Decorator for tracing async functions.
        
        Usage:
            @tracer.trace("my_function")
            async def my_function(prompt: str) -> str:
                ...
        
        Args:
            name: Trace name (defaults to function name)
            run_type: Type of run
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                
                # Build inputs from args/kwargs
                inputs = {
                    "args": self._truncate_str(str(args)[:500]),
                    "kwargs": self._truncate_str(str(kwargs)[:500])
                }
                
                async with self.trace_llm_call(trace_name, inputs, run_type=run_type) as trace:
                    result = await func(*args, **kwargs)
                    trace["output"] = result
                    return result
            
            return wrapper
        return decorator
    
    def _truncate_str(self, s: str, max_len: int = 2000) -> str:
        """Truncate string to max length."""
        if len(s) > max_len:
            return s[:max_len] + "... (truncated)"
        return s
    
    def _truncate_dict(self, d: Dict[str, Any], max_str_len: int = 1000) -> Dict[str, Any]:
        """Truncate string values in dict."""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = self._truncate_str(v, max_str_len)
            elif isinstance(v, dict):
                result[k] = self._truncate_dict(v, max_str_len)
            else:
                result[k] = v
        return result
    
    def _store_local_trace(self, trace: TraceMetadata):
        """Store trace locally for debugging when LangSmith is disabled."""
        self._local_traces[trace.run_id] = trace
        
        # Prune old traces
        if len(self._local_traces) > self._max_local_traces:
            oldest_keys = sorted(
                self._local_traces.keys(),
                key=lambda k: self._local_traces[k].start_time
            )[:100]
            for key in oldest_keys:
                del self._local_traces[key]
    
    def get_recent_traces(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Get recent traces for debugging."""
        traces = sorted(
            self._local_traces.values(),
            key=lambda t: t.start_time,
            reverse=True
        )[:limit]
        
        return [
            {
                "run_id": t.run_id,
                "name": t.name,
                "run_type": t.run_type,
                "start_time": t.start_time.isoformat(),
                "end_time": t.end_time.isoformat() if t.end_time else None,
                "duration_ms": t.duration_ms,
                "has_error": t.error is not None,
                "error": t.error
            }
            for t in traces
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        traces = list(self._local_traces.values())
        
        if not traces:
            return {
                "enabled": self.enabled,
                "total_traces": 0,
                "error_rate": 0,
                "avg_duration_ms": 0
            }
        
        errors = sum(1 for t in traces if t.error)
        durations = [t.duration_ms for t in traces if t.end_time]
        
        return {
            "enabled": self.enabled,
            "project_name": self.project_name,
            "total_traces": len(traces),
            "error_count": errors,
            "error_rate": errors / len(traces) if traces else 0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "traces_by_type": self._count_by_field(traces, "run_type"),
            "traces_by_name": self._count_by_field(traces, "name")
        }
    
    def _count_by_field(self, traces: list, field: str) -> Dict[str, int]:
        """Count traces by a field."""
        counts: Dict[str, int] = {}
        for t in traces:
            value = getattr(t, field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts


# Global tracer instance
langsmith_tracer = LangSmithTracer()
