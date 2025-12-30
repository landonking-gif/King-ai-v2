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
