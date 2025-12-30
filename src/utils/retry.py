"""
Retry utilities for resilient LLM calls and external API calls.
"""

import asyncio
import functools
from typing import Callable, Any
from src.utils.structured_logging import get_logger

logger = get_logger("retry")

# Default retry configuration for LLM calls
LLM_RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 1.0,  # seconds
    "max_delay": 10.0,  # seconds
    "exponential_base": 2
}


def with_retry(config: dict = None):
    """
    Decorator to retry async functions with exponential backoff.
    
    Args:
        config: Retry configuration dict with max_attempts, base_delay, etc.
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = LLM_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            max_attempts = config.get("max_attempts", 3)
            base_delay = config.get("base_delay", 1.0)
            max_delay = config.get("max_delay", 10.0)
            exp_base = config.get("exponential_base", 2)
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (exp_base ** attempt), max_delay)
                        
                        logger.warning(
                            "Retry attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=delay,
                            error=str(e)
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All retry attempts failed",
                            function=func.__name__,
                            max_attempts=max_attempts,
                            error=str(e)
                        )
            
            # Re-raise the last exception
            raise last_exception
        
        return wrapper
    
    return decorator
