"""
Retry utility - Provides decorators and utilities for retrying operations.
Useful for handling transient failures in API calls and network operations.
"""

import asyncio
import functools
from typing import Callable, TypeVar, Any
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2


def with_retry(config: RetryConfig):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        config: Retry configuration
        
    Example:
        @with_retry(RetryConfig(max_attempts=3))
        async def fetch_data():
            return await api.get("/data")
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        await asyncio.sleep(delay)
                    
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


# Pre-configured retry configurations for common use cases
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_base=2
)
