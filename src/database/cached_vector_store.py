"""
Cached Vector Store with Redis.

Implements a caching layer on top of Pinecone for reduced latency
and cost optimization. Uses Redis for fast local cache.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import redis.asyncio as redis

from config.settings import settings
from src.utils.structured_logging import get_logger
from src.database.vector_store import vector_store, MemoryType

logger = get_logger("cached_vector_store")


@dataclass
class CacheConfig:
    """Configuration for the caching layer."""
    # TTL for different memory types (in seconds)
    ttl_business_summary: int = 3600  # 1 hour
    ttl_conversation: int = 1800      # 30 minutes
    ttl_decision: int = 7200          # 2 hours
    ttl_task_result: int = 3600       # 1 hour
    ttl_market_research: int = 14400  # 4 hours
    ttl_financial_report: int = 7200  # 2 hours
    ttl_search_results: int = 900     # 15 minutes
    
    # Cache size limits
    max_cached_searches: int = 1000
    max_cached_memories: int = 5000


class CachedVectorStore:
    """
    Redis-cached wrapper around VectorStore.
    
    Implements two-level caching:
    1. Redis for fast distributed cache
    2. Pinecone for permanent storage
    
    Benefits:
    - Reduced Pinecone API calls (cost savings)
    - Lower latency for frequent queries
    - Resilience during Pinecone outages
    """
    
    # Cache key prefixes
    MEMORY_PREFIX = "kingai:memory:"
    SEARCH_PREFIX = "kingai:search:"
    STATS_KEY = "kingai:vector_cache:stats"
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "pinecone_calls": 0,
            "cache_writes": 0
        }
    
    async def connect(self):
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self._redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis cache: {e}")
            self._redis = None
            self._connected = False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    def _get_ttl(self, memory_type: str) -> int:
        """Get TTL for a memory type."""
        ttl_map = {
            MemoryType.BUSINESS_SUMMARY: self.config.ttl_business_summary,
            MemoryType.CONVERSATION: self.config.ttl_conversation,
            MemoryType.DECISION: self.config.ttl_decision,
            MemoryType.TASK_RESULT: self.config.ttl_task_result,
            MemoryType.MARKET_RESEARCH: self.config.ttl_market_research,
            MemoryType.FINANCIAL_REPORT: self.config.ttl_financial_report,
        }
        return ttl_map.get(memory_type, 3600)
    
    def _hash_query(self, query: str, memory_types: List[str] = None, top_k: int = 5) -> str:
        """Create a cache key for a search query."""
        key_data = f"{query}:{','.join(sorted(memory_types or []))}:{top_k}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def store_memory(
        self,
        text: str,
        memory_type: str,
        metadata: Dict[str, Any] = None,
        memory_id: str = None
    ) -> Optional[str]:
        """
        Store a memory with caching.
        
        Stores to both Redis cache and Pinecone.
        """
        # Store to Pinecone (primary storage)
        result_id = await vector_store.store_memory(
            text=text,
            memory_type=memory_type,
            metadata=metadata,
            memory_id=memory_id
        )
        
        if not result_id:
            return None
        
        # Cache in Redis
        if self._connected and self._redis:
            try:
                cache_key = f"{self.MEMORY_PREFIX}{result_id}"
                cache_data = {
                    "id": result_id,
                    "text": text,
                    "type": memory_type,
                    "metadata": metadata or {},
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                ttl = self._get_ttl(memory_type)
                await self._redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(cache_data)
                )
                self._stats["cache_writes"] += 1
                
                logger.debug(
                    "Cached memory",
                    memory_id=result_id,
                    ttl=ttl
                )
            except Exception as e:
                logger.warning(f"Failed to cache memory: {e}")
        
        return result_id
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID, checking cache first.
        """
        # Check cache first
        if self._connected and self._redis:
            try:
                cache_key = f"{self.MEMORY_PREFIX}{memory_id}"
                cached = await self._redis.get(cache_key)
                
                if cached:
                    self._stats["cache_hits"] += 1
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fall through to Pinecone
        self._stats["cache_misses"] += 1
        self._stats["pinecone_calls"] += 1
        
        # Note: vector_store doesn't have a direct get_by_id,
        # but we could add one or search with filter
        return None
    
    async def search_memories(
        self,
        query: str,
        memory_types: List[str] = None,
        top_k: int = 5,
        min_score: float = 0.7,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories with caching.
        
        Args:
            query: The search query
            memory_types: Filter by memory types
            top_k: Maximum results
            min_score: Minimum similarity score
            use_cache: Whether to use cache (default True)
            
        Returns:
            List of matching memories
        """
        # Check cache first
        if use_cache and self._connected and self._redis:
            try:
                query_hash = self._hash_query(query, memory_types, top_k)
                cache_key = f"{self.SEARCH_PREFIX}{query_hash}"
                
                cached = await self._redis.get(cache_key)
                if cached:
                    self._stats["cache_hits"] += 1
                    logger.debug("Search cache hit", query_hash=query_hash[:8])
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache search read failed: {e}")
        
        # Query Pinecone
        self._stats["cache_misses"] += 1
        self._stats["pinecone_calls"] += 1
        
        results = await vector_store.search_memories(
            query=query,
            memory_types=memory_types,
            top_k=top_k,
            min_score=min_score
        )
        
        # Cache the results
        if use_cache and self._connected and self._redis and results:
            try:
                query_hash = self._hash_query(query, memory_types, top_k)
                cache_key = f"{self.SEARCH_PREFIX}{query_hash}"
                
                await self._redis.setex(
                    cache_key,
                    self.config.ttl_search_results,
                    json.dumps(results)
                )
                self._stats["cache_writes"] += 1
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")
        
        return results
    
    async def invalidate_memory(self, memory_id: str):
        """Invalidate a cached memory."""
        if self._connected and self._redis:
            try:
                cache_key = f"{self.MEMORY_PREFIX}{memory_id}"
                await self._redis.delete(cache_key)
                logger.debug("Invalidated cached memory", memory_id=memory_id)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
    
    async def invalidate_searches(self):
        """Invalidate all cached searches (after new data is added)."""
        if self._connected and self._redis:
            try:
                # Use SCAN to find and delete search keys
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor,
                        match=f"{self.SEARCH_PREFIX}*",
                        count=100
                    )
                    if keys:
                        await self._redis.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                
                if deleted:
                    logger.info(f"Invalidated {deleted} cached searches")
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = (
            self._stats["cache_hits"] / total * 100
            if total > 0 else 0
        )
        
        cache_size = 0
        if self._connected and self._redis:
            try:
                # Count cached items
                info = await self._redis.info("keyspace")
                cache_size = sum(
                    v.get("keys", 0)
                    for v in info.values()
                    if isinstance(v, dict)
                )
            except Exception:
                pass
        
        return {
            "connected": self._connected,
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_writes": self._stats["cache_writes"],
            "pinecone_calls": self._stats["pinecone_calls"],
            "hit_rate_percent": round(hit_rate, 2),
            "estimated_cache_size": cache_size
        }
    
    async def warm_cache(self, business_ids: List[str] = None):
        """
        Pre-warm the cache with frequently accessed data.
        
        Args:
            business_ids: Optional list of business IDs to cache
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            logger.warning("Cannot warm cache - Redis not connected")
            return
        
        logger.info("Warming vector store cache")
        
        # Warm common searches
        common_queries = [
            "business performance summary",
            "recent decisions and outcomes",
            "market research insights",
            "financial projections"
        ]
        
        for query in common_queries:
            try:
                await self.search_memories(query, top_k=10)
            except Exception as e:
                logger.warning(f"Failed to warm query '{query}': {e}")
        
        logger.info("Cache warming complete")
    
    # Convenience methods that mirror vector_store
    async def store_business_summary(
        self,
        business_id: str,
        name: str,
        summary: str,
        kpis: Dict[str, Any] = None
    ) -> str:
        """Store a business summary with caching."""
        # Invalidate related searches
        await self.invalidate_searches()
        
        return await vector_store.store_business_summary(
            business_id=business_id,
            name=name,
            summary=summary,
            kpis=kpis
        )
    
    async def store_decision(
        self,
        decision: str,
        rationale: str,
        outcome: str = None,
        business_id: str = None
    ) -> str:
        """Store a decision with caching."""
        await self.invalidate_searches()
        
        return await vector_store.store_decision(
            decision=decision,
            rationale=rationale,
            outcome=outcome,
            business_id=business_id
        )
    
    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Get relevant context with caching."""
        memories = await self.search_memories(
            query=query,
            top_k=10,
            min_score=0.6
        )
        
        if not memories:
            return ""
        
        # Build context string
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for memory in memories:
            text = memory.get("text", "")
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(f"[{memory.get('type', 'unknown')}] {text}")
            total_chars += len(text)
        
        return "\n\n".join(context_parts)


# Global cached instance
cached_vector_store = CachedVectorStore()


async def initialize_cached_vector_store():
    """Initialize the cached vector store on startup."""
    await cached_vector_store.connect()
    logger.info("Cached vector store initialized")
