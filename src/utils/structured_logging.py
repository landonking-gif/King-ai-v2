"""
Structured logging module - backward compatibility wrapper.
Re-exports logging utilities for consistent imports.
"""

from src.utils.logging import get_logger, logger, setup_logging

__all__ = ["get_logger", "logger", "setup_logging"]
