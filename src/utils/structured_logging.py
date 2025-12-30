"""
Structured logging utility - Wrapper around structlog for consistent logging.
Provides a simple interface to get logger instances with context.
"""

import structlog
from contextvars import ContextVar

# Context variable for request-scoped logging
_request_context: ContextVar[dict] = ContextVar('request_context', default={})


def get_logger(name: str = None):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Structured logger with context binding
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


def set_request_context(**kwargs):
    """
    Set request-scoped context for logging.
    
    Args:
        **kwargs: Key-value pairs to add to request context
    """
    current = _request_context.get()
    _request_context.set({**current, **kwargs})


def get_request_context() -> dict:
    """Get the current request context."""
    return _request_context.get()


def clear_request_context():
    """Clear the request context."""
    _request_context.set({})
