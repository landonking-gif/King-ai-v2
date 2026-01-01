"""
LLM Response Cache.
Caches LLM responses to reduce costs and latency.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("llm_cache")


class CacheStrategy(str, Enum):
    """Cache strategies."""
    EXACT = "exact"  # Exact prompt match
    SEMANTIC = "semantic"  # Semantic similarity match
    HYBRID = "hybrid"  # Combine exact and semantic


class CacheStatus(str, Enum):
    """Cache entry status."""
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    default_ttl_seconds: int = 3600  # 1 hour
    max_entries: int = 10000
    strategy: CacheStrategy = CacheStrategy.EXACT
    similarity_threshold: float = 0.95  # For semantic matching
    model_specific_ttl: Dict[str, int] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass
class CacheEntry:
    """A cached LLM response."""
    key: str
    prompt_hash: str
    model: str
    prompt: str
    response: str
    tokens_used: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Update last accessed time and hit count."""
        self.hit_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    hits: int
    misses: int
    hit_rate: float
    tokens_saved: int
    cost_saved: float
    memory_used_mb: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_saved": self.tokens_saved,
            "cost_saved": round(self.cost_saved, 4),
            "memory_used_mb": round(self.memory_used_mb, 2),
            "oldest_entry": self.oldest_entry.isoformat() if self.oldest_entry else None,
            "newest_entry": self.newest_entry.isoformat() if self.newest_entry else None,
        }


class CacheKeyGenerator:
    """Generates cache keys from prompts."""
    
    @staticmethod
    def generate_exact(
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate exact match cache key."""
        content = json.dumps({
            "prompt": prompt,
            "model": model,
            "system": system_prompt,
            "temperature": temperature,
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def generate_normalized(
        prompt: str,
        model: str,
    ) -> str:
        """Generate normalized cache key (ignores whitespace)."""
        normalized = " ".join(prompt.split())
        content = f"{model}:{normalized}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def prompt_hash(prompt: str) -> str:
        """Generate prompt-only hash for similarity lookup."""
        return hashlib.md5(prompt.encode()).hexdigest()


class LRUEvictionPolicy:
    """Least Recently Used eviction policy."""
    
    def select_for_eviction(
        self,
        entries: Dict[str, CacheEntry],
        count: int,
    ) -> List[str]:
        """Select entries to evict."""
        # Sort by last accessed time
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].last_accessed,
        )
        
        return [key for key, _ in sorted_entries[:count]]


class LFUEvictionPolicy:
    """Least Frequently Used eviction policy."""
    
    def select_for_eviction(
        self,
        entries: Dict[str, CacheEntry],
        count: int,
    ) -> List[str]:
        """Select entries to evict."""
        # Sort by hit count, then by last accessed
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].hit_count, x[1].last_accessed),
        )
        
        return [key for key, _ in sorted_entries[:count]]


class LLMResponseCache:
    """
    Cache for LLM responses.
    
    Features:
    - Exact and semantic matching
    - TTL support
    - LRU/LFU eviction
    - Token savings tracking
    - Model-specific caching
    """
    
    # Estimated cost per token (USD)
    TOKEN_COSTS = {
        "gpt-4": 0.00003,
        "gpt-4-turbo": 0.00001,
        "gpt-3.5-turbo": 0.000001,
        "claude-3-opus": 0.000015,
        "claude-3-sonnet": 0.000003,
        "claude-3-haiku": 0.00000025,
    }
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._entries: Dict[str, CacheEntry] = {}
        self._prompt_index: Dict[str, List[str]] = {}  # prompt_hash -> [cache_keys]
        self._eviction_policy = LRUEvictionPolicy()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def get(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Tuple[Optional[str], CacheStatus]:
        """
        Get cached response for a prompt.
        
        Returns:
            Tuple of (response, status)
        """
        if not self.config.enabled:
            return None, CacheStatus.MISS
        
        async with self._lock:
            # Check for excluded patterns
            if self._is_excluded(prompt):
                self._misses += 1
                return None, CacheStatus.MISS
            
            # Generate cache key
            key = CacheKeyGenerator.generate_exact(
                prompt, model, system_prompt, temperature
            )
            
            # Look up entry
            entry = self._entries.get(key)
            
            if entry is None:
                self._misses += 1
                return None, CacheStatus.MISS
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None, CacheStatus.EXPIRED
            
            # Cache hit!
            entry.touch()
            self._hits += 1
            self._tokens_saved += entry.tokens_used
            
            logger.debug(f"Cache hit for {model}: {key[:16]}...")
            
            return entry.response, CacheStatus.HIT
    
    async def set(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_used: int,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        ttl_seconds: Optional[int] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Cache a response.
        
        Returns:
            Cache key
        """
        if not self.config.enabled:
            return ""
        
        async with self._lock:
            # Check for excluded patterns
            if self._is_excluded(prompt):
                return ""
            
            # Generate cache key
            key = CacheKeyGenerator.generate_exact(
                prompt, model, system_prompt, temperature
            )
            
            # Calculate TTL
            ttl = ttl_seconds or self.config.model_specific_ttl.get(
                model, self.config.default_ttl_seconds
            )
            
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None
            
            # Create entry
            entry = CacheEntry(
                key=key,
                prompt_hash=CacheKeyGenerator.prompt_hash(prompt),
                model=model,
                prompt=prompt,
                response=response,
                tokens_used=tokens_used,
                expires_at=expires_at,
                metadata=metadata or {},
            )
            
            # Evict if necessary
            if len(self._entries) >= self.config.max_entries:
                await self._evict(1)
            
            # Store entry
            self._entries[key] = entry
            
            # Update prompt index
            prompt_hash = entry.prompt_hash
            if prompt_hash not in self._prompt_index:
                self._prompt_index[prompt_hash] = []
            self._prompt_index[prompt_hash].append(key)
            
            logger.debug(f"Cached response for {model}: {key[:16]}...")
            
            return key
    
    async def invalidate(
        self,
        key: Optional[str] = None,
        model: Optional[str] = None,
        older_than: Optional[datetime] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            to_remove = []
            
            for entry_key, entry in self._entries.items():
                should_remove = False
                
                if key and entry_key == key:
                    should_remove = True
                elif model and entry.model == model:
                    should_remove = True
                elif older_than and entry.created_at < older_than:
                    should_remove = True
                
                if should_remove:
                    to_remove.append(entry_key)
            
            for entry_key in to_remove:
                self._remove_entry(entry_key)
            
            return len(to_remove)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._entries.clear()
            self._prompt_index.clear()
            logger.info("Cache cleared")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        entry = self._entries.pop(key, None)
        
        if entry:
            # Update prompt index
            if entry.prompt_hash in self._prompt_index:
                self._prompt_index[entry.prompt_hash] = [
                    k for k in self._prompt_index[entry.prompt_hash]
                    if k != key
                ]
    
    async def _evict(self, count: int) -> None:
        """Evict entries to make room."""
        keys_to_evict = self._eviction_policy.select_for_eviction(
            self._entries, count
        )
        
        for key in keys_to_evict:
            self._remove_entry(key)
        
        logger.debug(f"Evicted {len(keys_to_evict)} cache entries")
    
    def _is_excluded(self, prompt: str) -> bool:
        """Check if prompt matches exclusion patterns."""
        for pattern in self.config.exclude_patterns:
            if pattern.lower() in prompt.lower():
                return True
        return False
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        total_chars = sum(
            len(e.prompt) + len(e.response)
            for e in self._entries.values()
        )
        
        # Rough estimate: 2 bytes per character + overhead
        return (total_chars * 2 + len(self._entries) * 500) / (1024 * 1024)
    
    def _estimate_cost_saved(self) -> float:
        """Estimate cost saved from cache hits."""
        cost_saved = 0.0
        
        for entry in self._entries.values():
            if entry.hit_count > 0:
                cost_per_token = self.TOKEN_COSTS.get(entry.model, 0.00001)
                cost_saved += entry.tokens_used * cost_per_token * entry.hit_count
        
        return cost_saved
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        entries = list(self._entries.values())
        total = self._hits + self._misses
        
        oldest = min((e.created_at for e in entries), default=None)
        newest = max((e.created_at for e in entries), default=None)
        
        return CacheStats(
            total_entries=len(entries),
            hits=self._hits,
            misses=self._misses,
            hit_rate=self._hits / total if total > 0 else 0.0,
            tokens_saved=self._tokens_saved,
            cost_saved=self._estimate_cost_saved(),
            memory_used_mb=self._estimate_memory_mb(),
            oldest_entry=oldest,
            newest_entry=newest,
        )
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired = [
                key for key, entry in self._entries.items()
                if entry.is_expired
            ]
            
            for key in expired:
                self._remove_entry(key)
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired cache entries")
            
            return len(expired)
    
    async def warm_cache(
        self,
        prompts: List[Dict[str, Any]],
        llm_client: Any,
    ) -> int:
        """
        Pre-warm cache with common prompts.
        
        Args:
            prompts: List of prompt configs with 'prompt', 'model', etc.
            llm_client: LLM client to generate responses
            
        Returns:
            Number of prompts cached
        """
        cached = 0
        
        for prompt_config in prompts:
            prompt = prompt_config.get("prompt", "")
            model = prompt_config.get("model", "gpt-3.5-turbo")
            
            # Check if already cached
            response, status = await self.get(prompt, model)
            
            if status == CacheStatus.MISS:
                try:
                    # Generate response
                    result = await llm_client.generate(prompt, model=model)
                    
                    # Cache it
                    await self.set(
                        prompt=prompt,
                        response=result.get("content", ""),
                        model=model,
                        tokens_used=result.get("tokens", 0),
                    )
                    
                    cached += 1
                except Exception as e:
                    logger.error(f"Failed to warm cache: {e}")
        
        logger.info(f"Warmed cache with {cached} prompts")
        return cached


# Global cache instance
llm_cache = LLMResponseCache()


def get_llm_cache() -> LLMResponseCache:
    """Get the global LLM cache instance."""
    return llm_cache


# Decorator for caching LLM calls
def cached_llm_call(
    ttl_seconds: Optional[int] = None,
    exclude_temp_above: float = 0.5,
):
    """Decorator to cache LLM call results."""
    def decorator(func):
        async def wrapper(
            prompt: str,
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.0,
            **kwargs,
        ):
            # Skip cache for high temperature (non-deterministic)
            if temperature > exclude_temp_above:
                return await func(prompt, model=model, temperature=temperature, **kwargs)
            
            # Check cache
            cached_response, status = await llm_cache.get(
                prompt, model, temperature=temperature
            )
            
            if status == CacheStatus.HIT:
                return {"content": cached_response, "cached": True}
            
            # Call LLM
            result = await func(prompt, model=model, temperature=temperature, **kwargs)
            
            # Cache result
            await llm_cache.set(
                prompt=prompt,
                response=result.get("content", ""),
                model=model,
                tokens_used=result.get("tokens", 0),
                temperature=temperature,
                ttl_seconds=ttl_seconds,
            )
            
            return result
        
        return wrapper
    
    return decorator
