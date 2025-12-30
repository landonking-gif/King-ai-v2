"""
Structured logging for the Master AI system.
Provides context-aware logging with JSON output for aggregation.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps
import traceback

# Context variable for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='no-request-id')
user_id_var: ContextVar[str] = ContextVar('user_id', default='anonymous')
business_id_var: ContextVar[str] = ContextVar('business_id', default='none')


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for easy parsing."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "business_id": business_id_var.get(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Wrapper around Python's logging with structured output.
    Automatically includes context from ContextVars.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def _log(self, level: int, message: str, data: Dict[str, Any] = None, exc_info=None):
        """Internal logging method."""
        extra = {}
        if data:
            extra['extra_data'] = data
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs if kwargs else None)
    
    def error(self, message: str, exc_info=False, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, kwargs if kwargs else None, exc_info=exc_info)
    
    def critical(self, message: str, exc_info=False, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, kwargs if kwargs else None, exc_info=exc_info)
    
    def llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        error: str = None
    ):
        """Log an LLM API call."""
        self.info(
            "LLM call completed",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
    
    def agent_execution(
        self,
        agent: str,
        task: str,
        duration_ms: float,
        success: bool,
        error: str = None
    ):
        """Log an agent task execution."""
        self.info(
            "Agent execution completed",
            agent=agent,
            task=task,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def business_event(
        self,
        event_type: str,
        business_id: str,
        details: Dict[str, Any] = None
    ):
        """Log a business-related event."""
        self.info(
            f"Business event: {event_type}",
            event_type=event_type,
            business_id=business_id,
            details=details or {}
        )


def set_request_context(request_id: str, user_id: str = None, business_id: str = None):
    """Set context variables for the current request."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if business_id:
        business_id_var.set(business_id)


def log_function_call(logger: StructuredLogger):
    """Decorator to log function entry and exit."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}", success=True)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", exc_info=True, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}", success=True)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", exc_info=True, error=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# Create default loggers for each module
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module."""
    return StructuredLogger(f"king_ai.{name}")
