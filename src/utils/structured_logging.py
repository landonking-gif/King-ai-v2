"""
Structured logging utility - Wrapper around structlog for consistent logging.
Provides a simple interface to get logger instances with context.
"""

import structlog


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
