"""
Circuit Breaker Pattern Implementation.

Provides protection against cascading failures by temporarily
disabling calls to failing services.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, ParamSpec
from collections import deque

from src.utils.structured_logging import get_logger

logger = get_logger("circuit_breaker")

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, calls pass through
    OPEN = "open"          # Failures exceeded threshold, calls blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max calls allowed in half-open
    excluded_exceptions: tuple = field(default_factory=tuple)  # Don't count these as failures
    fallback: Optional[Callable] = None  # Fallback function when open


class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting service calls.
    
    Usage:
        cb = CircuitBreaker("external-api", failure_threshold=5, timeout=30)
        
        @cb.protect
        async def call_external_api():
            ...
            
        # Or manually:
        async with cb:
            result = await call_external_api()
    """
    
    # Global registry of circuit breakers for monitoring
    _registry: Dict[str, "CircuitBreaker"] = {}
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 30.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: tuple = (),
        fallback: Optional[Callable] = None,
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            half_open_max_calls=half_open_max_calls,
            excluded_exceptions=excluded_exceptions,
            fallback=fallback,
        )
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = datetime.utcnow()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        # Recent failures for analysis (keep last 100)
        self._recent_failures: deque = deque(maxlen=100)
        
        # Register for monitoring
        CircuitBreaker._registry[name] = self
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = (datetime.utcnow() - self._last_state_change).total_seconds()
            if elapsed >= self.config.timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get current statistics."""
        return self._stats
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()
        self._stats.state_changes += 1
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        logger.info(
            f"Circuit breaker state change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value
        )
    
    def _record_success(self):
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = datetime.utcnow()
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1
        
        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, exception: Exception):
        """Record a failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = datetime.utcnow()
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1
        
        self._recent_failures.append({
            "time": datetime.utcnow().isoformat(),
            "error": str(exception)[:200],
            "type": type(exception).__name__
        })
        
        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
    
    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self.state  # This triggers timeout check
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            return False
        elif state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            self._stats.rejected_calls += 1
            return False
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        async with self._lock:
            if not self._can_execute():
                raise CircuitOpenError(f"Circuit '{self.name}' is open")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        async with self._lock:
            if exc_type is None:
                self._record_success()
            elif exc_type and not issubclass(exc_type, self.config.excluded_exceptions):
                self._record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    def protect(self, func: Callable[P, T]) -> Callable[P, T]:
        """
        Decorator to protect an async function with this circuit breaker.
        
        Usage:
            @circuit_breaker.protect
            async def my_function():
                ...
        """
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with self._lock:
                if not self._can_execute():
                    if self.config.fallback:
                        return await self.config.fallback(*args, **kwargs)
                    raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
            try:
                result = await func(*args, **kwargs)
                async with self._lock:
                    self._record_success()
                return result
            except self.config.excluded_exceptions:
                async with self._lock:
                    self._record_success()  # Excluded exceptions count as success
                raise
            except Exception as e:
                async with self._lock:
                    self._record_failure(e)
                
                if self.config.fallback:
                    return await self.config.fallback(*args, **kwargs)
                raise
        
        return wrapper
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._stats = CircuitStats()
        self._recent_failures.clear()
        logger.info(f"Circuit breaker manually reset", name=self.name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "consecutive_failures": self._stats.consecutive_failures,
            "state_changes": self._stats.state_changes,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            "recent_failures": list(self._recent_failures)[-5:],
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout": self.config.timeout,
                "success_threshold": self.config.success_threshold,
            }
        }
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered circuit breakers."""
        return {name: cb.get_metrics() for name, cb in cls._registry.items()}


class CircuitOpenError(Exception):
    """Raised when attempting to call through an open circuit."""
    pass


# Pre-configured circuit breakers for common services
llm_circuit = CircuitBreaker(
    "llm",
    failure_threshold=3,
    timeout=60.0,
    success_threshold=2
)

shopify_circuit = CircuitBreaker(
    "shopify",
    failure_threshold=5,
    timeout=30.0,
    success_threshold=3
)

plaid_circuit = CircuitBreaker(
    "plaid",
    failure_threshold=5,
    timeout=30.0,
    success_threshold=3
)

stripe_circuit = CircuitBreaker(
    "stripe",
    failure_threshold=5,
    timeout=30.0,
    success_threshold=3
)

supplier_circuit = CircuitBreaker(
    "supplier",
    failure_threshold=5,
    timeout=60.0,  # Suppliers may be slower
    success_threshold=3
)


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
    success_threshold: int = 3,
    fallback: Optional[Callable] = None,
) -> Callable:
    """
    Decorator factory for creating protected functions.
    
    Usage:
        @circuit_breaker("my-service", failure_threshold=3)
        async def call_service():
            ...
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=success_threshold,
        fallback=fallback,
    )
    return cb.protect


# FastAPI dependency for health checks
async def get_circuit_breakers_health() -> Dict[str, Any]:
    """Get health status of all circuit breakers for API endpoint."""
    metrics = CircuitBreaker.get_all_metrics()
    
    unhealthy = [
        name for name, m in metrics.items()
        if m["state"] == CircuitState.OPEN.value
    ]
    
    return {
        "status": "unhealthy" if unhealthy else "healthy",
        "open_circuits": unhealthy,
        "circuits": metrics
    }
