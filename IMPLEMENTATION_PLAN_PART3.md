# King AI v2 - Implementation Plan Part 3 of 20
## Master AI Brain - Context & Memory System

**Target Timeline:** Week 3
**Objective:** Build a sophisticated context and memory system with RAG integration, semantic search, and intelligent token management.

---

## Table of Contents
1. [Overview of 20-Part Plan](#overview-of-20-part-plan)
2. [Part 3 Scope](#part-3-scope)
3. [Current State Analysis](#current-state-analysis)
4. [Implementation Tasks](#implementation-tasks)
5. [File-by-File Instructions](#file-by-file-instructions)
6. [Testing Requirements](#testing-requirements)
7. [Acceptance Criteria](#acceptance-criteria)

---

## Overview of 20-Part Plan

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| **3** | **Master AI Brain - Context & Memory System** | 🔄 Current |
| 4 | Master AI Brain - Planning & ReAct Implementation | ⏳ Pending |
| 5 | Evolution Engine - Code Modification System | ⏳ Pending |
| 6 | Evolution Engine - ML Retraining Pipeline | ⏳ Pending |
| 7 | Evolution Engine - Sandbox & Testing | ⏳ Pending |
| 8 | Sub-Agents - Research Agent Enhancement | ⏳ Pending |
| 9 | Sub-Agents - Code Generator Agent | ⏳ Pending |
| 10 | Sub-Agents - Content Agent | ⏳ Pending |
| 11 | Sub-Agents - Commerce Agent (Shopify/AliExpress) | ⏳ Pending |
| 12 | Sub-Agents - Finance Agent (Stripe/Plaid) | ⏳ Pending |
| 13 | Sub-Agents - Analytics Agent | ⏳ Pending |
| 14 | Sub-Agents - Legal Agent | ⏳ Pending |
| 15 | Business Units - Lifecycle Engine | ⏳ Pending |
| 16 | Business Units - Playbook System | ⏳ Pending |
| 17 | Business Units - Portfolio Management | ⏳ Pending |
| 18 | Dashboard - React UI Components | ⏳ Pending |
| 19 | Dashboard - Approval Workflows & Risk Engine | ⏳ Pending |
| 20 | Dashboard - Real-time Monitoring & WebSocket + Final Integration | ⏳ Pending |

---

## Part 3 Scope

This part focuses on:
1. Enhanced ContextManager with intelligent data gathering
2. RAG (Retrieval Augmented Generation) integration with Pinecone
3. Conversation history management with persistence
4. Token-aware context truncation and summarization
5. Semantic search for relevant business data
6. Long-term memory storage and retrieval
7. Embedding generation for vector storage

---

## Current State Analysis

### What Exists in `src/master_ai/context.py`
| Feature | Status | Issue |
|---------|--------|-------|
| Basic ContextManager class | ✅ Exists | Limited functionality |
| build_context method | ✅ Works | No RAG integration |
| Business summary | ✅ Basic | No semantic search |
| Task history | ✅ Basic | Limited to 20 tasks |
| Conversation history | ✅ Basic | No summarization |
| Token management | ⚠️ Crude | Simple character truncation |

### What Exists in `src/database/vector_store.py`
| Feature | Status | Issue |
|---------|--------|-------|
| Pinecone connection | ✅ Exists | Basic setup only |
| upsert_business_summary | ✅ Works | No embedding generation |
| search_similar_businesses | ✅ Basic | Not integrated with context |

### What Needs to Be Added
1. Embedding generation (via Ollama or external API)
2. RAG pipeline for context enrichment
3. Conversation summarization for long histories
4. Intelligent token budgeting per context section
5. Semantic memory storage and retrieval
6. Context prioritization based on relevance

---

## Implementation Tasks

### Task 3.1: Create Embedding Client
**Priority:** 🔴 Critical
**Estimated Time:** 3 hours
**Dependencies:** Part 1 complete

#### File: `src/utils/embedding_client.py` (CREATE NEW FILE)
```python
"""
Embedding generation client for vector operations.
Supports multiple providers: Ollama, OpenAI, or Sentence Transformers.
"""

import httpx
import asyncio
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

from config.settings import settings
from src.utils.structured_logging import get_logger

logger = get_logger("embeddings")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding generation using Ollama's embedding models.
    Uses models like nomic-embed-text or mxbai-embed-large.
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = "nomic-embed-text"
    ):
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
        self._dimension = 768  # Default for nomic-embed-text
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't support batch embedding natively, so we parallelize
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding using Sentence Transformers.
    Fallback when Ollama embeddings aren't available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # Default for MiniLM
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        model = self._get_model()
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()


class EmbeddingClient:
    """
    Unified embedding client with provider fallback.
    Tries Ollama first, falls back to local Sentence Transformers.
    """
    
    def __init__(self):
        self.primary: Optional[EmbeddingProvider] = None
        self.fallback: Optional[EmbeddingProvider] = None
        
        # Try to initialize Ollama provider
        try:
            self.primary = OllamaEmbeddingProvider()
            logger.info("Initialized Ollama embedding provider")
        except Exception as e:
            logger.warning(f"Ollama embeddings unavailable: {e}")
        
        # Initialize fallback
        try:
            self.fallback = SentenceTransformerProvider()
            logger.info("Initialized Sentence Transformer fallback")
        except Exception as e:
            logger.warning(f"Sentence Transformers unavailable: {e}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension from active provider."""
        if self.primary:
            return self.primary.dimension
        if self.fallback:
            return self.fallback.dimension
        return 768  # Default
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding with fallback."""
        if self.primary:
            try:
                return await self.primary.embed(text)
            except Exception as e:
                logger.warning(f"Primary embedding failed: {e}")
        
        if self.fallback:
            return await self.fallback.embed(text)
        
        raise RuntimeError("No embedding providers available")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with fallback."""
        if self.primary:
            try:
                return await self.primary.embed_batch(texts)
            except Exception as e:
                logger.warning(f"Primary batch embedding failed: {e}")
        
        if self.fallback:
            return await self.fallback.embed_batch(texts)
        
        raise RuntimeError("No embedding providers available")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


# Singleton instance
embedding_client = EmbeddingClient()
```

---

### Task 3.2: Enhance Vector Store with Full RAG Support
**Priority:** 🔴 Critical
**Estimated Time:** 3 hours
**Dependencies:** Task 3.1

#### File: `src/database/vector_store.py` (REPLACE ENTIRE FILE)
```python
"""
Enhanced Pinecone integration for RAG (Retrieval Augmented Generation).
Stores and retrieves business data embeddings for semantic search.
"""

from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from src.utils.embedding_client import embedding_client
from src.utils.structured_logging import get_logger

logger = get_logger("vector_store")


class MemoryType:
    """Types of memories stored in the vector database."""
    BUSINESS_SUMMARY = "business_summary"
    CONVERSATION = "conversation"
    DECISION = "decision"
    TASK_RESULT = "task_result"
    MARKET_RESEARCH = "market_research"
    FINANCIAL_REPORT = "financial_report"


class VectorStore:
    """
    Manages semantic storage and retrieval of business-related documents.
    Provides long-term memory for the Master AI.
    """
    
    def __init__(self):
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.index_name = settings.pinecone_index
        
        if settings.pinecone_api_key:
            try:
                self.pc = Pinecone(api_key=settings.pinecone_api_key)
                self._ensure_index_exists()
                self.index = self.pc.Index(self.index_name)
                logger.info("Vector store initialized", index=self.index_name)
            except Exception as e:
                logger.error("Failed to initialize vector store", error=str(e))
                self.pc = None
                self.index = None
        else:
            logger.warning("Pinecone API key not configured, vector store disabled")
    
    def _ensure_index_exists(self):
        """Create the index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info("Creating Pinecone index", index=self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=embedding_client.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment
                )
            )
    
    async def store_memory(
        self,
        text: str,
        memory_type: str,
        metadata: Dict[str, Any] = None,
        memory_id: str = None
    ) -> str:
        """
        Store a piece of information in long-term memory.
        
        Args:
            text: The text content to store
            memory_type: Type of memory (from MemoryType)
            metadata: Additional metadata to store
            memory_id: Optional ID (generated if not provided)
            
        Returns:
            The memory ID
        """
        if not self.index:
            logger.warning("Vector store not available, skipping memory storage")
            return None
        
        memory_id = memory_id or str(uuid4())
        
        try:
            # Generate embedding
            embedding = await embedding_client.embed(text)
            
            # Prepare metadata
            full_metadata = {
                "text": text[:1000],  # Pinecone metadata limit
                "full_text": text if len(text) <= 10000 else text[:10000],
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[{
                    "id": memory_id,
                    "values": embedding,
                    "metadata": full_metadata
                }]
            )
            
            logger.info(
                "Stored memory",
                memory_id=memory_id,
                type=memory_type,
                text_length=len(text)
            )
            
            return memory_id
            
        except Exception as e:
            logger.error("Failed to store memory", error=str(e))
            return None
    
    async def search_memories(
        self,
        query: str,
        memory_types: List[str] = None,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using semantic similarity.
        
        Args:
            query: The search query
            memory_types: Filter by memory types (optional)
            top_k: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of matching memories with scores
        """
        if not self.index:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await embedding_client.embed(query)
            
            # Build filter
            filter_dict = None
            if memory_types:
                filter_dict = {"type": {"$in": memory_types}}
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Process results
            memories = []
            for match in results.matches:
                if match.score >= min_score:
                    memories.append({
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("full_text", match.metadata.get("text", "")),
                        "type": match.metadata.get("type"),
                        "timestamp": match.metadata.get("timestamp"),
                        "metadata": {
                            k: v for k, v in match.metadata.items()
                            if k not in ["text", "full_text", "type", "timestamp"]
                        }
                    })
            
            logger.info(
                "Memory search completed",
                query_length=len(query),
                results=len(memories)
            )
            
            return memories
            
        except Exception as e:
            logger.error("Memory search failed", error=str(e))
            return []
    
    async def store_business_summary(
        self,
        business_id: str,
        name: str,
        summary: str,
        kpis: Dict[str, Any] = None
    ) -> str:
        """Store a business unit summary for later retrieval."""
        metadata = {
            "business_id": business_id,
            "business_name": name,
            "kpis": str(kpis) if kpis else ""
        }
        return await self.store_memory(
            text=f"Business: {name}\n{summary}",
            memory_type=MemoryType.BUSINESS_SUMMARY,
            metadata=metadata,
            memory_id=f"business_{business_id}"
        )
    
    async def store_conversation_summary(
        self,
        summary: str,
        turn_count: int,
        key_topics: List[str] = None
    ) -> str:
        """Store a summarized conversation for context."""
        metadata = {
            "turn_count": turn_count,
            "key_topics": ",".join(key_topics) if key_topics else ""
        }
        return await self.store_memory(
            text=summary,
            memory_type=MemoryType.CONVERSATION,
            metadata=metadata
        )
    
    async def store_decision(
        self,
        decision: str,
        rationale: str,
        outcome: str = None,
        business_id: str = None
    ) -> str:
        """Store a significant decision for learning."""
        text = f"Decision: {decision}\nRationale: {rationale}"
        if outcome:
            text += f"\nOutcome: {outcome}"
        
        metadata = {"business_id": business_id} if business_id else {}
        return await self.store_memory(
            text=text,
            memory_type=MemoryType.DECISION,
            metadata=metadata
        )
    
    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context from memory for a query.
        Combines multiple memory types into a coherent context string.
        
        Args:
            query: The query to find relevant context for
            max_tokens: Approximate maximum tokens for the result
            
        Returns:
            Formatted context string
        """
        memories = await self.search_memories(
            query=query,
            top_k=10,
            min_score=0.6
        )
        
        if not memories:
            return ""
        
        # Build context with token budget
        context_parts = []
        current_tokens = 0
        chars_per_token = 4  # Rough estimate
        
        for memory in memories:
            text = memory["text"]
            estimated_tokens = len(text) // chars_per_token
            
            if current_tokens + estimated_tokens > max_tokens:
                # Truncate this memory to fit
                remaining_chars = (max_tokens - current_tokens) * chars_per_token
                if remaining_chars > 100:
                    text = text[:remaining_chars] + "..."
                    context_parts.append(f"[{memory['type']}] {text}")
                break
            
            context_parts.append(f"[{memory['type']}] {text}")
            current_tokens += estimated_tokens
        
        return "\n\n".join(context_parts)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if not self.index:
            return False
        
        try:
            self.index.delete(ids=[memory_id])
            logger.info("Deleted memory", memory_id=memory_id)
            return True
        except Exception as e:
            logger.error("Failed to delete memory", error=str(e))
            return False
    
    async def clear_old_memories(
        self,
        memory_type: str,
        older_than_days: int = 30
    ) -> int:
        """Clear memories older than specified days."""
        # Note: Pinecone doesn't support date-based deletion directly
        # This would require iterating through memories
        # For now, log a warning
        logger.warning(
            "clear_old_memories not fully implemented",
            memory_type=memory_type,
            older_than_days=older_than_days
        )
        return 0


# Singleton instance
vector_store = VectorStore()
```

---

### Task 3.3: Create Token Manager for Context Budgeting
**Priority:** 🟡 High
**Estimated Time:** 2 hours
**Dependencies:** None

#### File: `src/utils/token_manager.py` (CREATE NEW FILE)
```python
"""
Token management utilities for context window optimization.
Handles token counting, budgeting, and intelligent truncation.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("token_manager")


class ContextSection(Enum):
    """Sections of the context window with priority levels."""
    SYSTEM_PROMPT = "system_prompt"           # Priority 1 (always include)
    CURRENT_STATE = "current_state"           # Priority 2
    RELEVANT_MEMORY = "relevant_memory"       # Priority 3
    RECENT_CONVERSATION = "recent_conversation"  # Priority 4
    BUSINESS_DATA = "business_data"           # Priority 5
    TASK_HISTORY = "task_history"             # Priority 6
    FULL_HISTORY = "full_history"             # Priority 7 (lowest)


@dataclass
class TokenBudget:
    """Token allocation for each context section."""
    section: ContextSection
    allocated: int
    used: int = 0
    priority: int = 0
    content: str = ""
    
    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)
    
    @property
    def utilization(self) -> float:
        return self.used / self.allocated if self.allocated > 0 else 0


@dataclass
class ContextBudget:
    """Complete context window budget allocation."""
    total_tokens: int
    sections: Dict[ContextSection, TokenBudget] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.sections:
            self._allocate_default_budget()
    
    def _allocate_default_budget(self):
        """Allocate default token budget based on 128K context window."""
        # Default allocation percentages
        allocations = {
            ContextSection.SYSTEM_PROMPT: (0.05, 1),       # 5% - always needed
            ContextSection.CURRENT_STATE: (0.15, 2),       # 15% - critical
            ContextSection.RELEVANT_MEMORY: (0.20, 3),     # 20% - RAG context
            ContextSection.RECENT_CONVERSATION: (0.15, 4), # 15% - recent turns
            ContextSection.BUSINESS_DATA: (0.20, 5),       # 20% - business info
            ContextSection.TASK_HISTORY: (0.15, 6),        # 15% - task context
            ContextSection.FULL_HISTORY: (0.10, 7),        # 10% - historical
        }
        
        for section, (percentage, priority) in allocations.items():
            self.sections[section] = TokenBudget(
                section=section,
                allocated=int(self.total_tokens * percentage),
                priority=priority
            )
    
    def get_budget(self, section: ContextSection) -> TokenBudget:
        """Get budget for a specific section."""
        return self.sections.get(section, TokenBudget(section=section, allocated=0))
    
    def set_content(self, section: ContextSection, content: str, tokens: int = None):
        """Set content for a section and update token usage."""
        if section not in self.sections:
            return
        
        budget = self.sections[section]
        budget.content = content
        budget.used = tokens if tokens is not None else estimate_tokens(content)
    
    def get_total_used(self) -> int:
        """Get total tokens used across all sections."""
        return sum(b.used for b in self.sections.values())
    
    def get_remaining(self) -> int:
        """Get remaining tokens in the total budget."""
        return max(0, self.total_tokens - self.get_total_used())
    
    def can_add(self, tokens: int) -> bool:
        """Check if we can add more tokens."""
        return self.get_remaining() >= tokens
    
    def build_context(self) -> str:
        """Build the final context string from all sections."""
        # Sort by priority and concatenate
        sorted_sections = sorted(
            self.sections.values(),
            key=lambda b: b.priority
        )
        
        parts = []
        for budget in sorted_sections:
            if budget.content:
                parts.append(budget.content)
        
        return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Uses rough heuristic of 4 characters per token for English text.
    """
    if not text:
        return 0
    return len(text) // 4


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken (more accurate but requires library).
    Falls back to estimation if tiktoken not available.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        return estimate_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int, preserve_end: bool = False) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        preserve_end: If True, keep the end of the text instead of the beginning
        
    Returns:
        Truncated text
    """
    current_tokens = estimate_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4
    
    if preserve_end:
        truncated = "..." + text[-char_limit:]
    else:
        truncated = text[:char_limit] + "..."
    
    return truncated


def smart_truncate(
    text: str,
    max_tokens: int,
    preserve_structure: bool = True
) -> str:
    """
    Intelligently truncate text while preserving structure.
    Tries to break at sentence or paragraph boundaries.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        preserve_structure: Try to break at natural boundaries
        
    Returns:
        Truncated text
    """
    current_tokens = estimate_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    if not preserve_structure:
        return truncate_to_tokens(text, max_tokens)
    
    # Try to break at paragraph boundaries
    paragraphs = text.split("\n\n")
    result = []
    total_tokens = 0
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if total_tokens + para_tokens <= max_tokens:
            result.append(para)
            total_tokens += para_tokens
        else:
            # Try to fit partial paragraph
            remaining = max_tokens - total_tokens
            if remaining > 50:  # Only if meaningful space left
                sentences = para.split(". ")
                for sent in sentences:
                    sent_tokens = estimate_tokens(sent)
                    if total_tokens + sent_tokens <= max_tokens:
                        result.append(sent + ".")
                        total_tokens += sent_tokens
                    else:
                        break
            break
    
    truncated = "\n\n".join(result)
    if len(truncated) < len(text):
        truncated += "\n\n[Content truncated...]"
    
    return truncated


class ConversationSummarizer:
    """Summarizes long conversations to fit in context window."""
    
    def __init__(self, llm_client):
        """
        Initialize with an LLM client for summarization.
        
        Args:
            llm_client: Client with a complete() method
        """
        self.llm = llm_client
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500
    ) -> str:
        """
        Summarize a conversation history.
        
        Args:
            messages: List of {role, content} message dicts
            max_tokens: Target token count for summary
            
        Returns:
            Summarized conversation
        """
        if not messages:
            return ""
        
        # Format messages for summarization
        formatted = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        ])
        
        prompt = f"""Summarize this conversation concisely, preserving key information and decisions:

{formatted}

Provide a summary in {max_tokens // 4} words or less, focusing on:
1. Main topics discussed
2. Decisions made
3. Action items or requests
4. Important context for future reference
"""
        
        try:
            summary = await self.llm.complete(prompt)
            return smart_truncate(summary, max_tokens)
        except Exception as e:
            logger.error("Conversation summarization failed", error=str(e))
            # Fallback: simple truncation
            return smart_truncate(formatted, max_tokens)
    
    async def create_rolling_summary(
        self,
        previous_summary: str,
        new_messages: List[Dict[str, str]],
        max_tokens: int = 500
    ) -> str:
        """
        Update a rolling summary with new messages.
        
        Args:
            previous_summary: Existing conversation summary
            new_messages: New messages to incorporate
            max_tokens: Target token count
            
        Returns:
            Updated summary
        """
        new_formatted = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in new_messages
        ])
        
        prompt = f"""Update this conversation summary with new messages.

PREVIOUS SUMMARY:
{previous_summary}

NEW MESSAGES:
{new_formatted}

Provide an updated summary in {max_tokens // 4} words or less that:
1. Incorporates new information
2. Maintains important context from before
3. Drops less relevant older details if needed
"""
        
        try:
            return await self.llm.complete(prompt)
        except Exception as e:
            logger.error("Rolling summary failed", error=str(e))
            return previous_summary
```

---

### Task 3.4: Enhanced Context Manager with Full RAG Integration
**Priority:** 🔴 Critical
**Estimated Time:** 4 hours
**Dependencies:** Tasks 3.1, 3.2, 3.3

#### File: `src/master_ai/context.py` (REPLACE ENTIRE FILE)
```python
"""
Enhanced Context Manager - Builds the full context window for the Master AI.

This module is responsible for:
1. Loading the current state from the database (Active Businesses, Tasks, History).
2. RAG integration for relevant memory retrieval.
3. Intelligent token budgeting and prioritization.
4. Conversation summarization for long histories.
5. Semantic search for related business context.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

from src.database.connection import get_db
from src.database.models import BusinessUnit, Task, ConversationMessage, Log
from src.database.vector_store import vector_store, MemoryType
from src.utils.token_manager import (
    ContextBudget, ContextSection, estimate_tokens,
    smart_truncate, ConversationSummarizer
)
from src.utils.structured_logging import get_logger
from config.settings import settings
from sqlalchemy import select

logger = get_logger("context_manager")


class ContextManager:
    """
    Manages the global state visibility for the MasterAI.
    Acts as a 'Sensory System' for the brain with RAG capabilities.
    """
    
    # Model context window limits
    CONTEXT_LIMITS = {
        "llama3.1:70b": 128000,
        "llama3.1:8b": 128000,
        "gemini-pro": 32000,
        "default": 100000
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize context manager.
        
        Args:
            llm_client: Optional LLM client for summarization
        """
        self.llm_client = llm_client
        self._summarizer = None
        self._conversation_cache: List[Dict[str, str]] = []
        self._rolling_summary: str = ""
        self._last_summary_turn = 0
        
        # Get context limit for current model
        model = settings.ollama_model
        self.max_context_tokens = self.CONTEXT_LIMITS.get(
            model, self.CONTEXT_LIMITS["default"]
        )
        
        # Reserve 20% for response
        self.available_tokens = int(self.max_context_tokens * 0.8)
        
        logger.info(
            "Context manager initialized",
            model=model,
            max_tokens=self.available_tokens
        )
    
    @property
    def summarizer(self) -> Optional[ConversationSummarizer]:
        """Lazy-load summarizer when LLM client is available."""
        if self._summarizer is None and self.llm_client:
            self._summarizer = ConversationSummarizer(self.llm_client)
        return self._summarizer
    
    async def build_context(
        self,
        query: str = None,
        include_rag: bool = True
    ) -> str:
        """
        Constructs a comprehensive snapshot of the entire empire.
        
        Args:
            query: Optional query for RAG-enhanced context
            include_rag: Whether to include semantic memory search
        
        Returns:
            Formatted context string optimized for token budget
        """
        logger.debug("Building context", query_provided=query is not None)
        
        # Initialize budget
        budget = ContextBudget(total_tokens=self.available_tokens)
        
        # Section 1: Current State (always include)
        current_state = await self._build_current_state()
        budget.set_content(ContextSection.CURRENT_STATE, current_state)
        
        # Section 2: Business Data
        business_data = await self._get_businesses_summary()
        budget.set_content(
            ContextSection.BUSINESS_DATA,
            smart_truncate(business_data, budget.get_budget(ContextSection.BUSINESS_DATA).allocated)
        )
        
        # Section 3: RAG - Relevant Memory (if query provided)
        if include_rag and query:
            relevant_memory = await self._get_relevant_memories(
                query,
                max_tokens=budget.get_budget(ContextSection.RELEVANT_MEMORY).allocated
            )
            budget.set_content(ContextSection.RELEVANT_MEMORY, relevant_memory)
        
        # Section 4: Recent Conversation
        conversation = await self._get_conversation_context(
            max_tokens=budget.get_budget(ContextSection.RECENT_CONVERSATION).allocated
        )
        budget.set_content(ContextSection.RECENT_CONVERSATION, conversation)
        
        # Section 5: Task History
        task_history = await self._get_recent_tasks()
        budget.set_content(
            ContextSection.TASK_HISTORY,
            smart_truncate(task_history, budget.get_budget(ContextSection.TASK_HISTORY).allocated)
        )
        
        # Section 6: Pending Approvals
        pending = await self._get_pending_approvals()
        # Add to current state if there's room
        if pending and budget.can_add(estimate_tokens(pending)):
            current = budget.get_budget(ContextSection.CURRENT_STATE).content
            budget.set_content(ContextSection.CURRENT_STATE, current + "\n\n" + pending)
        
        # Build final context
        full_context = budget.build_context()
        
        logger.info(
            "Context built",
            total_tokens=budget.get_total_used(),
            sections_used=len([s for s in budget.sections.values() if s.content])
        )
        
        return full_context
    
    async def _build_current_state(self) -> str:
        """Build the current system state section."""
        sections = [
            f"CURRENT TIME: {datetime.now().isoformat()}",
            f"RISK PROFILE: {settings.risk_profile}",
            f"AUTONOMOUS MODE: {'Enabled' if settings.enable_autonomous_mode else 'Disabled'}",
        ]
        
        # Add quick stats
        async with get_db() as db:
            result = await db.execute(select(BusinessUnit))
            businesses = result.scalars().all()
            
            if businesses:
                total_revenue = sum(b.total_revenue or 0 for b in businesses)
                total_expenses = sum(b.total_expenses or 0 for b in businesses)
                active_count = len([b for b in businesses if b.status.value != "sunset"])
                
                sections.append(f"EMPIRE OVERVIEW:")
                sections.append(f"  - Active Businesses: {active_count}")
                sections.append(f"  - Total Revenue: ${total_revenue:,.2f}")
                sections.append(f"  - Total Profit: ${total_revenue - total_expenses:,.2f}")
        
        return "\n".join(sections)

    async def _get_businesses_summary(self) -> str:
        """
        Fetches all active business units and summarizes their financial performance.
        """
        async with get_db() as db:
            result = await db.execute(
                select(BusinessUnit).order_by(BusinessUnit.created_at.desc())
            )
            businesses = result.scalars().all()
            
            if not businesses:
                return "BUSINESSES: None active. Ready to start new ventures."
            
            lines = ["BUSINESSES:"]
            for b in businesses:
                revenue = b.total_revenue or 0.0
                expenses = b.total_expenses or 0.0
                profit = revenue - expenses
                status = b.status.value if hasattr(b.status, 'value') else str(b.status)
                
                lines.append(
                    f"  - {b.name} ({b.type}): {status} | "
                    f"Revenue: ${revenue:,.2f} | Profit: ${profit:,.2f}"
                )
                
                # Add KPIs if available
                if b.kpis:
                    kpi_str = ", ".join([f"{k}: {v}" for k, v in list(b.kpis.items())[:3]])
                    lines.append(f"    KPIs: {kpi_str}")
            
            return "\n".join(lines)
    
    async def _get_relevant_memories(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Retrieve relevant context from vector store using RAG.
        """
        if not query:
            return ""
        
        try:
            context = await vector_store.get_relevant_context(
                query=query,
                max_tokens=max_tokens
            )
            
            if context:
                return f"RELEVANT CONTEXT FROM MEMORY:\n{context}"
            return ""
            
        except Exception as e:
            logger.warning("Failed to retrieve memories", error=str(e))
            return ""
    
    async def _get_conversation_context(self, max_tokens: int = 2000) -> str:
        """
        Get conversation history with intelligent summarization.
        """
        async with get_db() as db:
            result = await db.execute(
                select(ConversationMessage)
                .order_by(ConversationMessage.created_at.desc())
                .limit(50)
            )
            messages = result.scalars().all()
            
            if not messages:
                return ""
            
            # Reverse to chronological order
            messages = list(reversed(messages))
            
            # Check if we need summarization
            total_tokens = sum(estimate_tokens(m.content) for m in messages)
            
            if total_tokens <= max_tokens:
                # Can include full history
                lines = ["RECENT CONVERSATION:"]
                for m in messages[-20:]:  # Last 20 messages
                    role = m.role.upper()
                    content = m.content[:500]  # Truncate individual messages
                    lines.append(f"  {role}: {content}")
                return "\n".join(lines)
            
            # Need to summarize older messages
            if self.summarizer:
                # Split into old (to summarize) and recent (to keep full)
                split_point = len(messages) // 2
                old_messages = [{"role": m.role, "content": m.content} for m in messages[:split_point]]
                recent_messages = messages[split_point:]
                
                # Get or create summary of older messages
                if len(old_messages) > self._last_summary_turn + 10:
                    self._rolling_summary = await self.summarizer.summarize_conversation(
                        old_messages,
                        max_tokens=max_tokens // 3
                    )
                    self._last_summary_turn = len(old_messages)
                
                lines = ["CONVERSATION HISTORY:"]
                if self._rolling_summary:
                    lines.append(f"[Earlier summary]: {self._rolling_summary}")
                
                lines.append("\nRecent messages:")
                for m in recent_messages[-10:]:
                    role = m.role.upper()
                    content = m.content[:300]
                    lines.append(f"  {role}: {content}")
                
                return smart_truncate("\n".join(lines), max_tokens)
            
            # Fallback: simple truncation
            lines = ["RECENT CONVERSATION:"]
            for m in messages[-10:]:
                role = m.role.upper()
                content = m.content[:200]
                lines.append(f"  {role}: {content}")
            return "\n".join(lines)
    
    async def _get_recent_tasks(self) -> str:
        """
        Retrieves recent tasks to show what the system has been doing.
        """
        async with get_db() as db:
            result = await db.execute(
                select(Task).order_by(Task.created_at.desc()).limit(20)
            )
            tasks = result.scalars().all()
            
            if not tasks:
                return "RECENT TASKS: None"
            
            lines = ["RECENT TASKS:"]
            for t in tasks:
                status_icon = "✅" if t.status == "completed" else "⏳" if t.status == "pending" else "❌"
                lines.append(f"  {status_icon} {t.name} ({t.agent or 'system'}) - {t.status}")
            
            return "\n".join(lines)
    
    async def _get_pending_approvals(self) -> str:
        """Get pending items requiring human approval."""
        async with get_db() as db:
            result = await db.execute(
                select(Task)
                .where(Task.status == "pending_approval")
                .order_by(Task.created_at.desc())
            )
            pending = result.scalars().all()
            
            if not pending:
                return ""
            
            lines = ["⚠️ PENDING APPROVALS:"]
            for t in pending:
                lines.append(f"  - [{t.id[:8]}] {t.name}: {t.description or 'No description'}")
            
            return "\n".join(lines)
    
    async def add_to_conversation(
        self,
        role: str,
        content: str,
        persist: bool = True
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            persist: Whether to save to database
        """
        # Add to cache
        self._conversation_cache.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Persist to database
        if persist:
            async with get_db() as db:
                from uuid import uuid4
                msg = ConversationMessage(
                    id=str(uuid4()),
                    role=role,
                    content=content
                )
                db.add(msg)
                await db.commit()
        
        # Store significant exchanges in vector memory
        if len(self._conversation_cache) % 10 == 0:
            await self._store_conversation_to_memory()
    
    async def _store_conversation_to_memory(self):
        """Store conversation summary to long-term memory."""
        if not self._conversation_cache:
            return
        
        try:
            # Create summary for storage
            recent = self._conversation_cache[-10:]
            summary_text = "\n".join([
                f"{m['role']}: {m['content'][:200]}"
                for m in recent
            ])
            
            await vector_store.store_memory(
                text=f"Conversation on {datetime.now().isoformat()}:\n{summary_text}",
                memory_type=MemoryType.CONVERSATION,
                metadata={"turn_count": len(self._conversation_cache)}
            )
            
            logger.debug("Stored conversation to long-term memory")
            
        except Exception as e:
            logger.warning("Failed to store conversation", error=str(e))
    
    async def store_decision(
        self,
        decision: str,
        rationale: str,
        outcome: str = None,
        business_id: str = None
    ):
        """Store a significant decision in long-term memory."""
        await vector_store.store_decision(
            decision=decision,
            rationale=rationale,
            outcome=outcome,
            business_id=business_id
        )
    
    async def get_similar_decisions(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past decisions for reference."""
        return await vector_store.search_memories(
            query=query,
            memory_types=[MemoryType.DECISION],
            top_k=top_k
        )
    
    def clear_conversation_cache(self):
        """Clear the in-memory conversation cache."""
        self._conversation_cache = []
        self._rolling_summary = ""
        self._last_summary_turn = 0
```

---

## Testing Requirements

### Unit Tests

#### File: `tests/test_context_memory.py` (CREATE NEW FILE)
```python
"""
Tests for context and memory system.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.master_ai.context import ContextManager
from src.database.vector_store import VectorStore, MemoryType
from src.utils.token_manager import (
    ContextBudget, ContextSection, estimate_tokens,
    smart_truncate, TokenBudget
)
from src.utils.embedding_client import EmbeddingClient


class TestTokenManager:
    """Tests for token management utilities."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world this is a test"
        tokens = estimate_tokens(text)
        # Roughly 4 chars per token
        assert 5 <= tokens <= 10
    
    def test_context_budget_allocation(self):
        """Test budget is properly allocated."""
        budget = ContextBudget(total_tokens=10000)
        
        # Check all sections have allocations
        for section in ContextSection:
            assert budget.get_budget(section).allocated > 0
    
    def test_smart_truncate_preserves_structure(self):
        """Test smart truncation preserves paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        truncated = smart_truncate(text, max_tokens=20)
        
        # Should end at a paragraph boundary
        assert "paragraph" in truncated
    
    def test_budget_can_add(self):
        """Test budget capacity checking."""
        budget = ContextBudget(total_tokens=1000)
        
        assert budget.can_add(500)
        budget.set_content(ContextSection.CURRENT_STATE, "x" * 4000)  # 1000 tokens
        assert not budget.can_add(500)


class TestEmbeddingClient:
    """Tests for embedding generation."""
    
    @pytest.fixture
    def client(self):
        return EmbeddingClient()
    
    def test_cosine_similarity(self, client):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        similarity = client.cosine_similarity(a, b)
        assert similarity == pytest.approx(1.0)
    
    def test_cosine_similarity_orthogonal(self, client):
        """Test orthogonal vectors have 0 similarity."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        
        similarity = client.cosine_similarity(a, b)
        assert similarity == pytest.approx(0.0)


class TestVectorStore:
    """Tests for vector store operations."""
    
    @pytest.fixture
    def mock_store(self):
        with patch('src.database.vector_store.Pinecone'):
            with patch('src.database.vector_store.embedding_client') as mock_embed:
                mock_embed.embed = AsyncMock(return_value=[0.1] * 768)
                mock_embed.dimension = 768
                store = VectorStore()
                store.index = MagicMock()
                return store
    
    @pytest.mark.asyncio
    async def test_store_memory(self, mock_store):
        """Test storing a memory."""
        memory_id = await mock_store.store_memory(
            text="Test memory content",
            memory_type=MemoryType.DECISION,
            metadata={"key": "value"}
        )
        
        assert memory_id is not None
        mock_store.index.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_memories(self, mock_store):
        """Test searching memories."""
        mock_store.index.query.return_value = MagicMock(
            matches=[
                MagicMock(
                    id="test-id",
                    score=0.9,
                    metadata={"text": "Test", "type": "decision", "full_text": "Full test"}
                )
            ]
        )
        
        results = await mock_store.search_memories("test query")
        
        assert len(results) == 1
        assert results[0]["score"] == 0.9


class TestContextManager:
    """Tests for context manager."""
    
    @pytest.fixture
    def context_manager(self):
        return ContextManager()
    
    @pytest.mark.asyncio
    async def test_build_context_returns_string(self, context_manager):
        """Test that build_context returns a string."""
        with patch('src.master_ai.context.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_db.return_value.__aenter__.return_value = mock_session
            
            context = await context_manager.build_context()
            
            assert isinstance(context, str)
            assert "CURRENT TIME" in context
    
    @pytest.mark.asyncio
    async def test_add_to_conversation(self, context_manager):
        """Test adding messages to conversation."""
        with patch('src.master_ai.context.get_db') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            await context_manager.add_to_conversation(
                role="user",
                content="Hello",
                persist=True
            )
            
            assert len(context_manager._conversation_cache) == 1
            assert context_manager._conversation_cache[0]["role"] == "user"
```

---

## Acceptance Criteria

### Part 3 Completion Checklist

- [ ] **Embedding Client**
  - [ ] `src/utils/embedding_client.py` created
  - [ ] Ollama embedding provider working
  - [ ] Sentence Transformer fallback working
  - [ ] Cosine similarity calculation correct

- [ ] **Vector Store Enhanced**
  - [ ] `src/database/vector_store.py` updated
  - [ ] Memory storage working
  - [ ] Semantic search working
  - [ ] Multiple memory types supported
  - [ ] Context retrieval optimized

- [ ] **Token Manager**
  - [ ] `src/utils/token_manager.py` created
  - [ ] Token estimation working
  - [ ] Context budget allocation working
  - [ ] Smart truncation preserving structure
  - [ ] Conversation summarization working

- [ ] **Context Manager Enhanced**
  - [ ] `src/master_ai/context.py` updated
  - [ ] RAG integration working
  - [ ] Token budgeting enforced
  - [ ] Conversation history with summarization
  - [ ] Decision storage and retrieval

- [ ] **Tests Passing**
  - [ ] All unit tests pass
  - [ ] Memory operations tested
  - [ ] Context building tested

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/utils/embedding_client.py` |
| REPLACE | `src/database/vector_store.py` |
| CREATE | `src/utils/token_manager.py` |
| REPLACE | `src/master_ai/context.py` |
| CREATE | `tests/test_context_memory.py` |

---

## Dependencies to Install

Add these to `pyproject.toml`:
```toml
[project.dependencies]
# ... existing deps ...
sentence-transformers = "^2.2.0"  # Fallback embeddings
tiktoken = "^0.5.0"               # Accurate token counting (optional)
numpy = "^1.24.0"                 # Vector operations
```

---

## Environment Variables

Add to `.env`:
```bash
# Embedding Configuration
EMBEDDING_MODEL=nomic-embed-text    # Ollama embedding model
EMBEDDING_DIMENSION=768             # Must match model output

# Pinecone Configuration (if not already set)
PINECONE_API_KEY=your-api-key
PINECONE_INDEX=king-ai
PINECONE_ENV=us-east-1
```

---

## Next Part Preview

**Part 4: Master AI Brain - Planning & ReAct Implementation** will cover:
- Enhanced Planner with ReAct (Reason-Act-Think) pattern
- Multi-step goal decomposition
- Dependency-aware task scheduling
- Dynamic replanning on failure
- Execution monitoring and adjustment

---

*End of Part 3 - Master AI Brain - Context & Memory System*
