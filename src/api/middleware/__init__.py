"""
API Middleware Package.

Provides middleware components for:
- Rate limiting
- Authentication (JWT + API key)
- Logging
- Error handling
"""

from src.api.middleware.rate_limiter import (
    RateLimitMiddleware,
    RateLimitConfig,
    RateLimitDependency,
    InMemoryRateLimiter,
    RedisRateLimiter,
    create_rate_limit_middleware,
    RATE_LIMITS,
)
from src.api.middleware.auth import AuthMiddleware

__all__ = [
    "RateLimitMiddleware",
    "RateLimitConfig",
    "RateLimitDependency",
    "InMemoryRateLimiter",
    "RedisRateLimiter",
    "create_rate_limit_middleware",
    "RATE_LIMITS",
    "AuthMiddleware",
]
