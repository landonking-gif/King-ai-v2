"""
Retry utilities with exponential backoff.
Handles transient failures gracefully.
"""

import asyncio
import random
from typing import TypeVar, Callable, Any, Type, Tuple
from functools import wraps
from dataclasses import dataclass

from src.utils.structured_logging import get_logger

logger = get_logger("retry")

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay for next retry with exponential backoff."""
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    if config.jitter:
        # Add random jitter (±25%)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


async def retry_async(
    func: Callable[..., Any],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> Any:
    """
    Execute an async function with retry logic.
    
    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        config: Retry configuration
        **kwargs: Keyword arguments for the function
        
    Returns:
        The function result
        
    Raises:
        The last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{config.max_attempts}",
                    function=func.__name__,
                    error=str(e),
                    delay_seconds=delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All retry attempts exhausted",
                    function=func.__name__,
                    error=str(e),
                    attempts=config.max_attempts
                )
    
    raise last_exception


def with_retry(config: RetryConfig = None):
    """
    Decorator to add retry logic to async functions.
    
    Usage:
        @with_retry(RetryConfig(max_attempts=5))
        async def flaky_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""
    pass


class RateLimitError(RetryableError):
    """Raised when API rate limit is hit."""
    def __init__(self, retry_after: float = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}s")


class TransientError(RetryableError):
    """Raised for temporary failures (network, timeout, etc.)."""
    pass


class PermanentError(Exception):
    """Raised for errors that should not be retried."""
    pass


# Pre-configured retry configs for common scenarios
LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retryable_exceptions=(TransientError, RateLimitError, TimeoutError, ConnectionError)
)

DB_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions=(TransientError, ConnectionError)
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable_exceptions=(TransientError, RateLimitError, TimeoutError)
)
