"""
Redis Storage Adapter for Session Management.

Module: memory-service/service/storage/redis.py
"""

from typing import Any, Dict, Optional
import json
from datetime import timedelta

import anyio
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError

from ..config import Settings


class RedisAdapter:
    """Redis adapter for session storage and caching."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Redis adapter.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        self.pool = ConnectionPool.from_url(
            self.settings.redis_url, decode_responses=True, max_connections=10
        )
        self.client = Redis(connection_pool=self.pool)
        # Test connection
        await self.client.ping()

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.aclose()
        if self.pool:
            await self.pool.aclose()

    async def set_session_data(
        self, session_id: str, data: Dict[str, Any], ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Store session data with TTL.

        Args:
            session_id: Unique session identifier
            data: Session data to store
            ttl_hours: Time-to-live in hours (uses config default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        ttl = ttl_hours or self.settings.session_ttl_hours
        key = f"session:{session_id}"

        try:
            serialized = json.dumps(data)
            await self.client.setex(key, timedelta(hours=ttl), serialized)
            return True
        except (RedisError, json.JSONDecodeError) as e:
            print(f"Error setting session data: {e}")
            return False

    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data.

        Args:
            session_id: Unique session identifier

        Returns:
            Session data if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"session:{session_id}"

        try:
            data = await self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            print(f"Error getting session data: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session data.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"session:{session_id}"

        try:
            result = await self.client.delete(key)
            return result > 0
        except RedisError as e:
            print(f"Error deleting session: {e}")
            return False

    async def extend_session_ttl(self, session_id: str, ttl_hours: int) -> bool:
        """
        Extend session TTL.

        Args:
            session_id: Unique session identifier
            ttl_hours: New TTL in hours

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"session:{session_id}"

        try:
            result = await self.client.expire(key, timedelta(hours=ttl_hours))
            return bool(result)
        except RedisError as e:
            print(f"Error extending session TTL: {e}")
            return False

    async def cache_artifact(
        self, artifact_id: str, content: Dict[str, Any], ttl_seconds: int = 3600
    ) -> bool:
        """
        Cache artifact content for fast retrieval.

        Args:
            artifact_id: Artifact identifier
            content: Artifact content
            ttl_seconds: Cache TTL in seconds

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"artifact:cache:{artifact_id}"

        try:
            serialized = json.dumps(content)
            await self.client.setex(key, timedelta(seconds=ttl_seconds), serialized)
            return True
        except (RedisError, json.JSONDecodeError) as e:
            print(f"Error caching artifact: {e}")
            return False

    async def get_cached_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached artifact.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Cached artifact if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"artifact:cache:{artifact_id}"

        try:
            data = await self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            print(f"Error getting cached artifact: {e}")
            return None

    async def add_to_session_artifacts(self, session_id: str, artifact_id: str) -> bool:
        """
        Add artifact ID to session's artifact list.

        Args:
            session_id: Session identifier
            artifact_id: Artifact identifier

        Returns:
            True if added successfully, False otherwise
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"session:artifacts:{session_id}"

        try:
            await self.client.sadd(key, artifact_id)
            # Set TTL on the set
            await self.client.expire(key, timedelta(hours=self.settings.session_ttl_hours))
            return True
        except RedisError as e:
            print(f"Error adding artifact to session: {e}")
            return False

    async def get_session_artifacts(self, session_id: str) -> list[str]:
        """
        Get all artifact IDs for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of artifact IDs
        """
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"session:artifacts:{session_id}"

        try:
            members = await self.client.smembers(key)
            return list(members)
        except RedisError as e:
            print(f"Error getting session artifacts: {e}")
            return []
