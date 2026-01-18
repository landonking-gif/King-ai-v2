"""
Tier 3 Memory - Long-term Vector Memory.

Stores important information with vector embeddings for semantic search.
This enables retrieval of relevant context from the entire history.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import numpy as np

from src.utils.structured_logging import get_logger

logger = get_logger("tier3_memory")


class MemoryCategory(str, Enum):
    """Categories for long-term memories."""
    DECISION = "decision"       # Important decisions made
    FINDING = "finding"         # Key research findings
    PREFERENCE = "preference"   # User preferences learned
    PROCESS = "process"         # Process/workflow information
    ENTITY = "entity"          # Important entities mentioned
    CODE = "code"              # Code snippets
    ERROR = "error"            # Errors and solutions
    OTHER = "other"


@dataclass
class LongTermMemory:
    """A single long-term memory entry with embedding."""
    
    id: str
    project_id: str
    content: str
    category: MemoryCategory = MemoryCategory.OTHER
    
    # Embedding and search
    embedding: Optional[List[float]] = None
    importance: float = 0.5  # 0.0 to 1.0
    
    # Metadata
    source: str = ""  # Where this memory came from
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Usage tracking
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "content": self.content,
            "category": self.category.value,
            "embedding": self.embedding,
            "importance": self.importance,
            "source": self.source,
            "session_id": self.session_id,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermMemory":
        data = data.copy()
        if "category" in data and isinstance(data["category"], str):
            data["category"] = MemoryCategory(data["category"])
        for dt_field in ["last_accessed", "created_at"]:
            if dt_field in data and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**data)


# LLM prompt for extracting memories
EXTRACTION_PROMPT = """Extract important information worth remembering long-term from this content.
Focus on: decisions made, key findings, user preferences, process information, and important entities.

Content:
{content}

Return a JSON object:
{{
    "memories": [
        {{
            "content": "A concise statement of the important information",
            "category": "decision|finding|preference|process|entity|code|error|other",
            "importance": 0.0-1.0
        }}
    ]
}}

Only include truly important information. Return empty array if nothing worth remembering.
Return ONLY valid JSON."""


class Tier3Memory:
    """
    Tier 3 Memory - Long-term Vector Memory.
    
    Stores important information with embeddings for semantic retrieval.
    Uses vector similarity search to find relevant context.
    
    Features:
    - LLM-powered memory extraction
    - Vector embeddings for semantic search
    - Importance-based ranking
    - Access count decay
    """
    
    EMBEDDING_DIM = 384  # Default for small embedding models
    
    def __init__(
        self,
        llm_client=None,
        embedding_client=None,
        redis_client=None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize Tier 3 memory.
        
        Args:
            llm_client: LLM for memory extraction
            embedding_client: Client for generating embeddings
            redis_client: Redis for vector storage
            storage_path: File-based storage path
        """
        self.llm = llm_client
        self.embedding_client = embedding_client
        self.redis = redis_client
        self.storage_path = storage_path or Path("data/memory/tier3")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for simple similarity search
        self._memory_index: Dict[str, List[LongTermMemory]] = {}
    
    async def store_memory(
        self,
        project_id: str,
        content: str,
        category: MemoryCategory = MemoryCategory.OTHER,
        source: str = "",
        session_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> LongTermMemory:
        """
        Store a single memory with embedding.
        
        Args:
            project_id: Project/business identifier
            content: Memory content
            category: Memory category
            source: Where this memory came from
            session_id: Associated session
            importance: Importance score (0-1)
            tags: Additional tags
            
        Returns:
            Stored memory
        """
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        memory = LongTermMemory(
            id=f"mem_{hashlib.sha256(content.encode()).hexdigest()[:16]}",
            project_id=project_id,
            content=content,
            category=category,
            embedding=embedding,
            importance=importance,
            source=source,
            session_id=session_id,
            tags=tags or [],
        )
        
        await self._store(project_id, memory)
        
        logger.debug(
            "Stored long-term memory",
            project_id=project_id,
            category=category.value,
            importance=importance,
        )
        
        return memory
    
    async def extract_and_store(
        self,
        project_id: str,
        content: str,
        source: str = "",
        session_id: Optional[str] = None
    ) -> List[LongTermMemory]:
        """
        Extract important memories from content and store them.
        
        Uses LLM to identify what's worth remembering.
        
        Args:
            project_id: Project identifier
            content: Content to extract memories from
            source: Source of the content
            session_id: Associated session
            
        Returns:
            List of extracted and stored memories
        """
        extracted = await self._extract_memories(content)
        
        memories = []
        for item in extracted:
            try:
                category = MemoryCategory(item.get("category", "other"))
            except ValueError:
                category = MemoryCategory.OTHER
            
            memory = await self.store_memory(
                project_id=project_id,
                content=item.get("content", ""),
                category=category,
                source=source,
                session_id=session_id,
                importance=float(item.get("importance", 0.5)),
            )
            memories.append(memory)
        
        logger.info(
            "Extracted and stored memories",
            project_id=project_id,
            count=len(memories),
        )
        
        return memories
    
    async def search(
        self,
        project_id: str,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        categories: Optional[List[MemoryCategory]] = None
    ) -> List[Tuple[LongTermMemory, float]]:
        """
        Search memories by semantic similarity.
        
        Args:
            project_id: Project identifier
            query: Search query
            limit: Maximum results
            min_importance: Minimum importance threshold
            categories: Filter by categories
            
        Returns:
            List of (memory, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Load all memories for project
        memories = await self._load_memories(project_id)
        
        # Calculate similarities
        results = []
        for memory in memories:
            # Apply filters
            if memory.importance < min_importance:
                continue
            if categories and memory.category not in categories:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            # Boost by importance
            score = similarity * (0.5 + 0.5 * memory.importance)
            
            results.append((memory, score))
        
        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]
        
        # Update access counts
        for memory, _ in results:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
        
        return results
    
    async def get_context_for_query(
        self,
        project_id: str,
        query: str,
        max_tokens: int = 1000
    ) -> str:
        """
        Get relevant context for a query as formatted string.
        
        Args:
            project_id: Project identifier
            query: The query to find context for
            max_tokens: Approximate max tokens to return
            
        Returns:
            Formatted context string
        """
        results = await self.search(project_id, query, limit=10)
        
        if not results:
            return ""
        
        lines = ["Relevant memories:"]
        total_chars = 0
        max_chars = max_tokens * 4  # Rough char estimate
        
        for memory, score in results:
            if total_chars > max_chars:
                break
            
            line = f"  [{memory.category.value}] {memory.content}"
            lines.append(line)
            total_chars += len(line)
        
        return "\n".join(lines)
    
    async def get_by_category(
        self,
        project_id: str,
        category: MemoryCategory,
        limit: int = 20
    ) -> List[LongTermMemory]:
        """Get memories by category."""
        memories = await self._load_memories(project_id)
        
        filtered = [m for m in memories if m.category == category]
        filtered.sort(key=lambda m: m.importance, reverse=True)
        
        return filtered[:limit]
    
    async def delete_memory(
        self,
        project_id: str,
        memory_id: str
    ) -> bool:
        """Delete a specific memory."""
        memories = await self._load_memories(project_id)
        
        original_count = len(memories)
        memories = [m for m in memories if m.id != memory_id]
        
        if len(memories) < original_count:
            await self._save_memories(project_id, memories)
            return True
        
        return False
    
    async def decay_old_memories(
        self,
        project_id: str,
        days_threshold: int = 30,
        decay_factor: float = 0.9
    ) -> int:
        """
        Apply decay to old, unused memories.
        
        Reduces importance of memories not accessed recently.
        
        Returns:
            Number of memories decayed
        """
        memories = await self._load_memories(project_id)
        threshold = datetime.utcnow().timestamp() - (days_threshold * 86400)
        
        decayed_count = 0
        for memory in memories:
            if memory.last_accessed.timestamp() < threshold:
                memory.importance *= decay_factor
                decayed_count += 1
        
        if decayed_count > 0:
            await self._save_memories(project_id, memories)
        
        return decayed_count
    
    # Private methods
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.embedding_client:
            try:
                return await self.embedding_client.embed(text)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
        
        # Fallback: simple hash-based pseudo-embedding
        # This is not semantically meaningful but allows basic operation
        return self._hash_embedding(text)
    
    def _hash_embedding(self, text: str) -> List[float]:
        """Generate a pseudo-embedding from text hash."""
        # Create deterministic pseudo-random embedding from text
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Convert to floats in [-1, 1]
        embedding = []
        for i in range(0, min(len(text_hash), self.EMBEDDING_DIM), 1):
            val = (text_hash[i % len(text_hash)] / 127.5) - 1.0
            embedding.append(val)
        
        # Pad to full dimension
        while len(embedding) < self.EMBEDDING_DIM:
            embedding.append(0.0)
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(
        self,
        vec1: Optional[List[float]],
        vec2: Optional[List[float]]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot / (norm_a * norm_b))
        except Exception:
            return 0.0
    
    async def _extract_memories(self, content: str) -> List[Dict[str, Any]]:
        """Extract important memories from content using LLM."""
        if not self.llm:
            return []
        
        prompt = EXTRACTION_PROMPT.format(content=content[:2000])
        
        try:
            response = await self.llm.complete(prompt)
            
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("memories", [])
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")
        
        return []
    
    async def _store(self, project_id: str, memory: LongTermMemory) -> None:
        """Store a memory."""
        memories = await self._load_memories(project_id)
        
        # Check for duplicate (same content)
        for existing in memories:
            if existing.id == memory.id:
                # Update existing
                existing.importance = max(existing.importance, memory.importance)
                existing.access_count += 1
                await self._save_memories(project_id, memories)
                return
        
        memories.append(memory)
        await self._save_memories(project_id, memories)
        self._memory_index[project_id] = memories
    
    async def _load_memories(self, project_id: str) -> List[LongTermMemory]:
        """Load memories from cache or storage."""
        if project_id in self._memory_index:
            return self._memory_index[project_id]
        
        memories = []
        
        if self.redis:
            # Load from Redis
            keys = await self.redis.keys(f"tier3:{project_id}:*")
            for key in keys:
                data = await self.redis.get(key)
                if data:
                    memories.append(LongTermMemory.from_dict(json.loads(data)))
        else:
            file_path = self.storage_path / f"{project_id}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    memories = [LongTermMemory.from_dict(m) for m in data]
        
        self._memory_index[project_id] = memories
        return memories
    
    async def _save_memories(
        self,
        project_id: str,
        memories: List[LongTermMemory]
    ) -> None:
        """Persist memories to storage."""
        if self.redis:
            # Store each memory separately in Redis
            for memory in memories:
                await self.redis.set(
                    f"tier3:{project_id}:{memory.id}",
                    json.dumps(memory.to_dict()),
                    ex=86400 * 365  # 1 year expiry
                )
        else:
            file_path = self.storage_path / f"{project_id}.json"
            
            # Don't store embeddings in file (too large)
            data = []
            for m in memories:
                d = m.to_dict()
                d["embedding"] = None  # Clear embedding for file storage
                data.append(d)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
