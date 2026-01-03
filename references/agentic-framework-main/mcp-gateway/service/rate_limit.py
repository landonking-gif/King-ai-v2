"""
Rate limiting implementation using Redis with in-memory fallback.

Provides token-bucket based rate limiting for tool invocations with per-tool and per-actor limits.
Falls back to in-memory rate limiting when Redis is not available.
"""

from typing import Optional, Dict, List
import redis.asyncio as redis
from datetime import datetime
import anyio
import time

from .config import settings
from .models import RateLimitConfig


class InMemoryRateLimiter:
    """Simple in-memory rate limiter for when Redis is unavailable."""

    def __init__(self) -> None:
        """Initialize in-memory rate limiter."""
        # Dict of key -> list of timestamps
        self._buckets: Dict[str, List[float]] = {}

    def _get_key(self, tool_id: str, actor_id: str) -> str:
        """Generate key for rate limit tracking."""
        return f"rate_limit:{tool_id}:{actor_id}"

    async def check_and_increment(
        self, tool_id: str, actor_id: str, rate_limit: RateLimitConfig
    ) -> tuple[bool, Optional[int]]:
        """Check if request is within rate limit."""
        key = self._get_key(tool_id, actor_id)
        now = time.time()
        window_start = now - rate_limit.window_seconds

        # Get or create bucket
        if key not in self._buckets:
            self._buckets[key] = []

        # Clean old entries
        self._buckets[key] = [t for t in self._buckets[key] if t > window_start]

        # Check limit
        if len(self._buckets[key]) >= rate_limit.max_calls:
            oldest = min(self._buckets[key]) if self._buckets[key] else now
            retry_after = int(oldest + rate_limit.window_seconds - now) + 1
            return False, retry_after

        # Add current request
        self._buckets[key].append(now)
        return True, None


class RateLimiter:
    """Redis-based rate limiter with in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        """
        Initialize rate limiter with Redis connection.

        Args:
            redis_url: Redis connection URL (uses settings default if not provided)
        """
        self.redis_url = redis_url or settings.redis_url
        self._client: Optional[redis.Redis] = None
        self._redis_available: bool = True
        self._fallback = InMemoryRateLimiter()

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None and self._redis_available:
            try:
                self._client = await redis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                # Test connection
                await self._client.ping()
            except Exception as e:
                print(f"[RATE_LIMIT] Redis unavailable, using in-memory fallback: {e}")
                self._redis_available = False
                self._client = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_key(self, tool_id: str, actor_id: str) -> str:
        """
        Generate Redis key for rate limit tracking.

        Args:
            tool_id: Tool identifier
            actor_id: Actor identifier

        Returns:
            Redis key string
        """
        return f"rate_limit:{tool_id}:{actor_id}"

    async def check_and_increment(
        self, tool_id: str, actor_id: str, rate_limit: RateLimitConfig
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is within rate limit and increment counter.

        Uses sliding window algorithm with Redis sorted sets.
        Falls back to in-memory rate limiting if Redis is unavailable.

        Args:
            tool_id: Tool identifier
            actor_id: Actor identifier
            rate_limit: Rate limit configuration

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        if self._client is None:
            await self.connect()

        # Use fallback if Redis not available
        if not self._redis_available or self._client is None:
            return await self._fallback.check_and_increment(tool_id, actor_id, rate_limit)

        assert self._client is not None  # for mypy

        key = self._get_key(tool_id, actor_id)
        now = datetime.utcnow().timestamp()
        window_start = now - rate_limit.window_seconds

        # Use Redis pipeline for atomic operations
        async with self._client.pipeline(transaction=True) as pipe:
            # Remove old entries outside the window
            await pipe.zremrangebyscore(key, 0, window_start)

            # Count current entries in window
            await pipe.zcard(key)

            # Add current request
            await pipe.zadd(key, {str(now): now})

            # Set expiration on key
            await pipe.expire(key, rate_limit.window_seconds)

            results = await pipe.execute()

        # results[1] contains the count before adding current request
        current_count = int(results[1])

        if current_count >= rate_limit.max_calls:
            # Get oldest entry to calculate retry_after
            oldest_entries = await self._client.zrange(key, 0, 0, withscores=True)
            if oldest_entries:
                oldest_timestamp = float(oldest_entries[0][1])
                retry_after = int(oldest_timestamp + rate_limit.window_seconds - now) + 1
                return False, retry_after

            return False, rate_limit.window_seconds

        return True, None

    async def get_current_usage(
        self, tool_id: str, actor_id: str, window_seconds: int
    ) -> int:
        """
        Get current usage count for a tool/actor combination.

        Args:
            tool_id: Tool identifier
            actor_id: Actor identifier
            window_seconds: Time window in seconds

        Returns:
            Current count of requests in the window
        """
        if self._client is None:
            await self.connect()

        assert self._client is not None  # for mypy

        key = self._get_key(tool_id, actor_id)
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds

        # Count entries in current window
        count = await self._client.zcount(key, window_start, now)
        return int(count)

    async def reset(self, tool_id: str, actor_id: str) -> None:
        """
        Reset rate limit for a specific tool/actor combination.

        Args:
            tool_id: Tool identifier
            actor_id: Actor identifier
        """
        if self._client is None:
            await self.connect()

        assert self._client is not None  # for mypy

        key = self._get_key(tool_id, actor_id)
        await self._client.delete(key)


# Global rate limiter instance
_rate_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance (singleton pattern).

    Returns:
        Global RateLimiter instance
    """
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter()
    return _rate_limiter_instance
