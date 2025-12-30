"""
Retry utilities - Provides retry decorators for LLM and other operations.
"""

import asyncio
import functools
from typing import Callable, Any, Dict
from dataclasses import dataclass

from src.utils.structured_logging import get_logger

logger = get_logger("retry")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    exceptions: tuple = (Exception,)


# Standard retry config for LLM operations
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    exceptions=(Exception,)
)


def with_retry(config: RetryConfig = None):
    """
    Decorator that adds retry logic to async functions.
    
    Args:
        config: RetryConfig instance, defaults to standard config
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = config.initial_delay
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}",
                            func=func.__name__,
                            delay=delay
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * config.exponential_base, config.max_delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed",
                            func=func.__name__,
                            error=str(e)
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def sync_with_retry(config: RetryConfig = None):
    """
    Decorator that adds retry logic to synchronous functions.
    
    Args:
        config: RetryConfig instance, defaults to standard config
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            delay = config.initial_delay
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}",
                            func=func.__name__,
                            delay=delay
                        )
                        time.sleep(delay)
                        delay = min(delay * config.exponential_base, config.max_delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed",
                            func=func.__name__,
                            error=str(e)
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator
