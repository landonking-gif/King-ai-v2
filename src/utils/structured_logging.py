"""
Alias module for structured logging.
Provides backward compatibility with the 'structured_logging' import pattern.
"""

from src.utils.logging import setup_logging, get_logger, logger

__all__ = ['setup_logging', 'get_logger', 'logger']
