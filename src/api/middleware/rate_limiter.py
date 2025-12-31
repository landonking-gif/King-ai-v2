"""
Rate Limiting Middleware for FastAPI.

Provides flexible rate limiting with:
- Per-endpoint limits
- Per-user/API key limits
- Sliding window algorithm
- Redis-backed distributed limiting
- Graceful degradation
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import defaultdict

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.structured_logging import get_logger

logger = get_logger("rate_limiter")


class RateLimitStrategy(str, Enum):
    """Rate limiting algorithms."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    key_prefix: str = ""  # Prefix for cache keys
    skip_successful_requests: bool = False  # Only count failed requests
    include_response_headers: bool = True  # Add X-RateLimit-* headers
    

@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float  # Unix timestamp
    limit: int
    retry_after: Optional[int] = None  # Seconds until retry


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.
    
    Suitable for single-instance deployments or development.
    For production, use RedisRateLimiter.
    """
    
    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
    
    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier (IP, user ID, API key hash)
            config: Rate limit configuration
        
        Returns:
            RateLimitResult with allowed status and metadata
        """
        async with self._locks[key]:
            now = time.time()
            window_start = now - config.window_seconds
            
            # Clean old entries
            self._windows[key] = [
                ts for ts in self._windows[key]
                if ts > window_start
            ]
            
            current_count = len(self._windows[key])
            
            if current_count >= config.requests:
                # Rate limit exceeded
                oldest = self._windows[key][0] if self._windows[key] else now
                retry_after = int(oldest + config.window_seconds - now) + 1
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest + config.window_seconds,
                    limit=config.requests,
                    retry_after=retry_after,
                )
            
            # Record this request
            self._windows[key].append(now)
            
            # Periodic cleanup
            await self._maybe_cleanup()
            
            return RateLimitResult(
                allowed=True,
                remaining=config.requests - current_count - 1,
                reset_at=now + config.window_seconds,
                limit=config.requests,
            )
    
    async def _maybe_cleanup(self):
        """Periodically clean up old entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        cutoff = now - 3600  # Clean entries older than 1 hour
        
        keys_to_remove = []
        for key, timestamps in self._windows.items():
            self._windows[key] = [ts for ts in timestamps if ts > cutoff]
            if not self._windows[key]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._windows[key]
            if key in self._locks:
                del self._locks[key]


class RedisRateLimiter:
    """
    Redis-backed rate limiter for distributed deployments.
    
    Uses Redis sorted sets for sliding window algorithm
    with atomic operations.
    """
    
    def __init__(self, redis_url: str = None):
        self._redis = None
        self._redis_url = redis_url
        self._fallback = InMemoryRateLimiter()
    
    async def _get_redis(self):
        """Lazy initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    self._redis_url or "redis://localhost:6379",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using fallback")
                return None
        return self._redis
    
    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit using Redis."""
        redis = await self._get_redis()
        
        if redis is None:
            # Fallback to in-memory
            return await self._fallback.check(key, config)
        
        try:
            now = time.time()
            window_start = now - config.window_seconds
            
            cache_key = f"ratelimit:{config.key_prefix}:{key}"
            
            # Use pipeline for atomic operations
            async with redis.pipeline(transaction=True) as pipe:
                # Remove old entries
                await pipe.zremrangebyscore(cache_key, 0, window_start)
                # Count current entries
                await pipe.zcard(cache_key)
                # Add new entry
                await pipe.zadd(cache_key, {str(now): now})
                # Set expiry
                await pipe.expire(cache_key, config.window_seconds + 10)
                
                results = await pipe.execute()
            
            current_count = results[1]
            
            if current_count >= config.requests:
                # Get oldest entry to calculate retry time
                oldest_entries = await redis.zrange(cache_key, 0, 0, withscores=True)
                oldest = oldest_entries[0][1] if oldest_entries else now
                retry_after = int(oldest + config.window_seconds - now) + 1
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest + config.window_seconds,
                    limit=config.requests,
                    retry_after=max(1, retry_after),
                )
            
            return RateLimitResult(
                allowed=True,
                remaining=config.requests - current_count - 1,
                reset_at=now + config.window_seconds,
                limit=config.requests,
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return await self._fallback.check(key, config)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Usage:
        app.add_middleware(
            RateLimitMiddleware,
            default_limit=RateLimitConfig(requests=100, window_seconds=60),
            limiter=RedisRateLimiter()
        )
    """
    
    def __init__(
        self,
        app,
        default_limit: RateLimitConfig = None,
        limiter: Union[InMemoryRateLimiter, RedisRateLimiter] = None,
        key_func: Callable[[Request], str] = None,
        skip_paths: list[str] = None,
        endpoint_limits: Dict[str, RateLimitConfig] = None,
    ):
        super().__init__(app)
        
        self.default_limit = default_limit or RateLimitConfig(
            requests=100,
            window_seconds=60
        )
        self.limiter = limiter or InMemoryRateLimiter()
        self.key_func = key_func or self._default_key_func
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.endpoint_limits = endpoint_limits or {}
    
    def _default_key_func(self, request: Request) -> str:
        """Generate rate limit key from request."""
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Try to get user from JWT (if authenticated)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _get_endpoint_config(self, request: Request) -> RateLimitConfig:
        """Get rate limit config for endpoint."""
        path = request.url.path
        method = request.method
        
        # Check exact match
        key = f"{method}:{path}"
        if key in self.endpoint_limits:
            return self.endpoint_limits[key]
        
        # Check path-only match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check pattern matches
        for pattern, config in self.endpoint_limits.items():
            if pattern.endswith("*") and path.startswith(pattern[:-1]):
                return config
        
        return self.default_limit
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Get rate limit key and config
        key = self.key_func(request)
        config = self._get_endpoint_config(request)
        
        # Check rate limit
        result = await self.limiter.check(key, config)
        
        if not result.allowed:
            logger.warning(
                "Rate limit exceeded",
                key=key,
                path=request.url.path,
                limit=result.limit,
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please retry after {result.retry_after} seconds.",
                    "retry_after": result.retry_after,
                },
                headers={
                    "Retry-After": str(result.retry_after),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers if configured
        if config.include_response_headers:
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))
        
        return response


# Dependency for per-endpoint rate limiting
class RateLimitDependency:
    """
    FastAPI dependency for fine-grained rate limiting.
    
    Usage:
        @app.get("/expensive-operation")
        async def expensive_op(
            _: None = Depends(RateLimitDependency(requests=10, window_seconds=60))
        ):
            ...
    """
    
    _limiter = InMemoryRateLimiter()
    
    def __init__(
        self,
        requests: int,
        window_seconds: int,
        key_func: Callable[[Request], str] = None,
        error_message: str = None,
    ):
        self.config = RateLimitConfig(
            requests=requests,
            window_seconds=window_seconds
        )
        self.key_func = key_func
        self.error_message = error_message or "Rate limit exceeded for this operation"
    
    async def __call__(self, request: Request):
        """Check rate limit."""
        if self.key_func:
            key = self.key_func(request)
        else:
            # Use client IP + endpoint as key
            ip = request.client.host if request.client else "unknown"
            key = f"{ip}:{request.url.path}"
        
        result = await self._limiter.check(key, self.config)
        
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": self.error_message,
                    "retry_after": result.retry_after,
                },
                headers={
                    "Retry-After": str(result.retry_after),
                }
            )
        
        return result


# Pre-configured rate limits for common patterns
RATE_LIMITS = {
    # Authentication endpoints - strict limits
    "POST:/api/v1/auth/login": RateLimitConfig(requests=5, window_seconds=60),
    "POST:/api/v1/auth/register": RateLimitConfig(requests=3, window_seconds=300),
    
    # API endpoints - moderate limits
    "/api/v1/businesses*": RateLimitConfig(requests=100, window_seconds=60),
    "/api/v1/approvals*": RateLimitConfig(requests=50, window_seconds=60),
    
    # LLM endpoints - conservative limits (expensive)
    "POST:/api/v1/chat": RateLimitConfig(requests=20, window_seconds=60),
    "POST:/api/v1/agents/execute": RateLimitConfig(requests=30, window_seconds=60),
    
    # Webhook endpoints - higher limits
    "POST:/api/v1/webhooks/*": RateLimitConfig(requests=200, window_seconds=60),
    
    # Export endpoints - strict limits
    "/api/v1/export/*": RateLimitConfig(requests=5, window_seconds=300),
}


def create_rate_limit_middleware(
    redis_url: str = None,
    default_requests: int = 100,
    default_window: int = 60,
) -> RateLimitMiddleware:
    """
    Factory function to create configured rate limit middleware.
    
    Args:
        redis_url: Redis URL for distributed limiting (optional)
        default_requests: Default requests per window
        default_window: Default window in seconds
    
    Returns:
        Configured RateLimitMiddleware
    """
    if redis_url:
        limiter = RedisRateLimiter(redis_url)
    else:
        limiter = InMemoryRateLimiter()
    
    return lambda app: RateLimitMiddleware(
        app,
        default_limit=RateLimitConfig(
            requests=default_requests,
            window_seconds=default_window
        ),
        limiter=limiter,
        endpoint_limits=RATE_LIMITS,
    )
