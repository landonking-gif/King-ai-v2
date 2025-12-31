"""
Structured logging utility - Wrapper around structlog for consistent logging.
Provides a simple interface to get logger instances with context.
"""

import structlog
from contextvars import ContextVar
from typing import Optional

# Context variable for request-scoped logging
_request_context: ContextVar[dict] = ContextVar('request_context', default={})


class EnhancedLogger:
    """
    Enhanced logger wrapper that adds custom logging methods.
    Wraps structlog's bound logger with additional methods.
    """
    
    def __init__(self, logger, name: str = None):
        self._logger = logger
        self._name = name
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped logger."""
        return getattr(self._logger, name)
    
    def llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Log an LLM API call with structured data.
        
        Args:
            provider: LLM provider (e.g., "ollama", "openai")
            model: Model name used
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency_ms: Call latency in milliseconds
            success: Whether the call succeeded
            error: Error message if failed
        """
        # NOTE: structlog reserves the key "event" for the log message.
        # Passing an "event" kwarg causes "multiple values for argument 'event'".
        event_data = {
            "event_type": "llm_call",
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "success": success
        }
        if error:
            event_data["error"] = error
        
        if success:
            self._logger.info("LLM call completed", **event_data)
        else:
            self._logger.error("LLM call failed", **event_data)


def get_logger(name: str = None):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Enhanced structured logger with context binding
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return EnhancedLogger(logger, name)


def set_request_context(*args, **kwargs):
    """
    Set request-scoped context for logging.
    
    Args:
        *args: Optional positional request_id (legacy)
        **kwargs: Key-value pairs to add to request context
    """
    if args:
        if len(args) != 1:
            raise TypeError("set_request_context accepts at most 1 positional argument")
        kwargs = {"request_id": args[0], **kwargs}

    current = _request_context.get()
    _request_context.set({**current, **kwargs})


def get_request_context() -> dict:
    """Get the current request context."""
    return _request_context.get()


def clear_request_context():
    """Clear the request context."""
    _request_context.set({})
