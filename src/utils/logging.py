"""
Structured Logging - Provides JSON-formatted logs for better observability.
Integrates with structlog for rich context and performance.
"""

import structlog
import logging
import sys

def setup_logging():
    """
    Configures structlog to output JSON logs to stdout.
    Used in API lifespan and background scripts.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

def get_logger(name: str = None):
    """
    Get a logger instance with an optional name.
    
    Args:
        name: Optional name for the logger (e.g., module name)
        
    Returns:
        A bound structlog logger instance
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()

# Global logger instance
logger = structlog.get_logger()

def get_logger(name: str = None):
    """
    Get a logger instance with optional name.
    
    Args:
        name: Optional logger name for context
        
    Returns:
        Structured logger instance
    """
    if name:
        return structlog.get_logger().bind(component=name)
    return structlog.get_logger()
