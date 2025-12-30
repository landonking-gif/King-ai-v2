"""
Helper function to get structured loggers.
"""

import structlog


def get_logger(name: str):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
