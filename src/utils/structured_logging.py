"""
Structured Logging - Wrapper around structlog for consistent logging.
"""

import structlog


def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
