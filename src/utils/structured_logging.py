"""
Structured Logging - Wrapper for getting logger instances with structured logging.
"""

import structlog
import logging


def get_logger(name: str):
    """
    Get a structured logger instance.
    
    Args:
        name: Name of the logger (typically the module name)
        
    Returns:
        A structlog logger instance
    """
    return structlog.get_logger(name)


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
        level=logging.INFO,
    )
