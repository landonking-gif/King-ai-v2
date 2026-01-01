"""
Agent Retry Decorator.
Provides automatic retry logic with circuit breaker integration for agent LLM calls.
"""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Coroutine, Dict, List, Optional, 
    Type, TypeVar, Union, ParamSpec
)
from enum import Enum

from src.utils.structured_logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker, CircuitState

logger = get_logger("agent_retry")

P = ParamSpec('P')
T = TypeVar('T')


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.25
    
    # Exceptions to retry on
    retryable_exceptions: tuple = (
        TimeoutError,
        ConnectionError,
        asyncio.TimeoutError,
    )
    
    # Exceptions to never retry
    non_retryable_exceptions: tuple = (
        ValueError,
        TypeError,
        KeyError,
    )
    
    # Fallback provider if all retries fail
    fallback_provider: Optional[str] = None


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    total_retries: int = 0
    fallback_calls: int = 0
    avg_attempts: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class AgentRetryHandler:
    """
    Handles retry logic for agent LLM calls.
    
    Features:
    - Multiple retry strategies (exponential, linear, etc.)
    - Circuit breaker integration
    - Fallback provider support
    - Detailed metrics
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.config = config or RetryConfig()
        self.circuit_breaker = circuit_breaker
        self.stats = RetryStats()
        self._attempt_counts: List[int] = []
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (2 ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            fibs = [1, 1]
            for _ in range(attempt):
                fibs.append(fibs[-1] + fibs[-2])
            delay = self.config.base_delay * fibs[min(attempt, len(fibs) - 1)]
        
        else:
            delay = self.config.base_delay
        
        # Apply max delay cap
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter
        if self.config.jitter:
            import random
            jitter = delay * self.config.jitter_factor * random.random()
            delay = delay + jitter
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Never retry these
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Always retry these
        if isinstance(exception, self.config.retryable_exceptions):
            return True
        
        # Check for specific error messages
        error_msg = str(exception).lower()
        retryable_messages = [
            "rate limit",
            "too many requests",
            "timeout",
            "connection refused",
            "service unavailable",
            "temporarily unavailable",
            "503",
            "429",
            "overloaded",
        ]
        
        return any(msg in error_msg for msg in retryable_messages)
    
    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        fallback_func: Optional[Callable[..., Coroutine[Any, Any, T]]] = None,
        **kwargs,
    ) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            fallback_func: Optional fallback function
            **kwargs: Keyword arguments
            
        Returns:
            Result from the function
            
        Raises:
            Last exception if all retries fail
        """
        self.stats.total_calls += 1
        last_exception: Optional[Exception] = None
        attempts = 0
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.state == CircuitState.OPEN:
            logger.warning("Circuit breaker is open, trying fallback")
            if fallback_func:
                self.stats.fallback_calls += 1
                return await fallback_func(*args, **kwargs)
            raise Exception("Circuit breaker is open and no fallback available")
        
        for attempt in range(self.config.max_attempts):
            attempts = attempt + 1
            
            try:
                result = await func(*args, **kwargs)
                
                # Success
                self.stats.successful_calls += 1
                self._attempt_counts.append(attempts)
                self._update_avg_attempts()
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                self.stats.last_error = str(e)
                self.stats.last_error_time = datetime.utcnow()
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if we should retry
                if not self._should_retry(e):
                    logger.warning(
                        f"Non-retryable exception: {type(e).__name__}",
                        error=str(e),
                    )
                    break
                
                # Check if we have attempts left
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.stats.retried_calls += 1
                    self.stats.total_retries += 1
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_attempts}",
                        delay=delay,
                        error=str(e),
                    )
                    
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self.stats.failed_calls += 1
        
        # Try fallback
        if fallback_func:
            logger.info("All retries exhausted, using fallback")
            self.stats.fallback_calls += 1
            try:
                return await fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error
        
        # No fallback, raise the last exception
        if last_exception:
            raise last_exception
        
        raise Exception("Unknown error during retry")
    
    def _update_avg_attempts(self) -> None:
        """Update average attempts metric."""
        if self._attempt_counts:
            # Keep last 100 for rolling average
            if len(self._attempt_counts) > 100:
                self._attempt_counts.pop(0)
            self.stats.avg_attempts = sum(self._attempt_counts) / len(self._attempt_counts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "retried_calls": self.stats.retried_calls,
            "total_retries": self.stats.total_retries,
            "fallback_calls": self.stats.fallback_calls,
            "avg_attempts": round(self.stats.avg_attempts, 2),
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls * 100
                if self.stats.total_calls > 0 else 0
            ),
            "last_error": self.stats.last_error,
            "last_error_time": (
                self.stats.last_error_time.isoformat()
                if self.stats.last_error_time else None
            ),
        }


def agent_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    fallback_provider: Optional[str] = None,
    circuit_breaker_name: Optional[str] = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator for agent methods with automatic retry logic.
    
    Usage:
        class MyAgent(BaseAgent):
            @agent_retry(max_attempts=3, fallback_provider="gemini")
            async def process(self, prompt: str) -> str:
                return await self.llm.generate(prompt)
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
        fallback_provider=fallback_provider,
    )
    
    # Get circuit breaker if specified
    circuit_breaker = None
    if circuit_breaker_name:
        try:
            from src.utils.circuit_breaker import get_breaker
            circuit_breaker = get_breaker(circuit_breaker_name)
        except Exception:
            pass
    
    handler = AgentRetryHandler(config=config, circuit_breaker=circuit_breaker)
    
    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await handler.execute(func, *args, **kwargs)
        
        # Attach stats method
        wrapper.get_retry_stats = handler.get_stats
        return wrapper
    
    return decorator


def with_fallback(
    fallback_func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator to add a fallback function if the primary fails.
    
    Usage:
        async def fallback_process(prompt: str) -> str:
            return await gemini_client.generate(prompt)
        
        class MyAgent(BaseAgent):
            @with_fallback(fallback_process)
            async def process(self, prompt: str) -> str:
                return await self.claude.generate(prompt)
    """
    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary failed, using fallback: {e}")
                return await fallback_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Convenience function for one-off retry operations
async def retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    **kwargs,
) -> T:
    """
    Execute an async function with retry logic.
    
    Usage:
        result = await retry_async(
            some_async_function,
            arg1, arg2,
            max_attempts=3,
            delay=1.0,
        )
    """
    handler = AgentRetryHandler(
        config=RetryConfig(
            max_attempts=max_attempts,
            base_delay=delay,
        )
    )
    return await handler.execute(func, *args, **kwargs)
