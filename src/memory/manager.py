"""
Memory Manager.

Unified interface for the 3-tier memory system.
Handles automatic tiering and context retrieval.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from src.memory.tier1_recent import Tier1Memory, RecentMessage
from src.memory.tier2_summaries import Tier2Memory, SessionSummary
from src.memory.tier3_longterm import Tier3Memory, LongTermMemory, MemoryCategory
from src.utils.structured_logging import get_logger

logger = get_logger("memory_manager")


class MemoryManager:
    """
    Unified interface for the 3-tier memory system.
    
    Provides:
    - Automatic tiering of messages
    - Unified context retrieval
    - Memory compression and summarization
    - Semantic search across all tiers
    """
    
    def __init__(
        self,
        llm_client=None,
        embedding_client=None,
        redis_client=None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            llm_client: LLM for summarization and extraction
            embedding_client: Client for generating embeddings
            redis_client: Redis for persistence
            storage_path: Base path for file storage
        """
        base_path = storage_path or Path("data/memory")
        
        self.tier1 = Tier1Memory(
            redis_client=redis_client,
            storage_path=base_path / "tier1",
        )
        
        self.tier2 = Tier2Memory(
            llm_client=llm_client,
            redis_client=redis_client,
            storage_path=base_path / "tier2",
        )
        
        self.tier3 = Tier3Memory(
            llm_client=llm_client,
            embedding_client=embedding_client,
            redis_client=redis_client,
            storage_path=base_path / "tier3",
        )
        
        self.llm = llm_client
    
    # Message handling
    
    async def add_user_message(
        self,
        session_id: str,
        content: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecentMessage:
        """Add a user message to Tier 1."""
        message = await self.tier1.add_user_message(session_id, content, metadata)
        
        # Check if we should extract long-term memories
        if project_id and len(content) > 100:
            # Extract important info to Tier 3 in background
            await self._maybe_extract_memories(project_id, content, session_id)
        
        return message
    
    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        project_id: Optional[str] = None,
        agent_invoked: Optional[str] = None,
        tool_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecentMessage:
        """Add an assistant message to Tier 1."""
        message = await self.tier1.add_assistant_message(
            session_id, content, agent_invoked, tool_used, metadata
        )
        
        # Check if we should extract long-term memories
        if project_id and len(content) > 200:
            await self._maybe_extract_memories(project_id, content, session_id)
        
        return message
    
    # Context retrieval
    
    async def get_context(
        self,
        session_id: str,
        project_id: Optional[str] = None,
        query: Optional[str] = None,
        include_tier1: bool = True,
        include_tier2: bool = True,
        include_tier3: bool = True,
        max_tokens: int = 4000
    ) -> str:
        """
        Get unified context from all memory tiers.
        
        Args:
            session_id: Current session
            project_id: Project for Tier 2/3 context
            query: Query for semantic search (Tier 3)
            include_tier1: Include recent messages
            include_tier2: Include session summaries
            include_tier3: Include semantic memories
            max_tokens: Approximate max tokens
            
        Returns:
            Formatted context string
        """
        sections = []
        remaining_tokens = max_tokens
        
        # Tier 1: Recent context (prioritize this)
        if include_tier1:
            tier1_tokens = min(remaining_tokens // 2, 2000)
            tier1_context = await self.tier1.get_context_string(session_id)
            if tier1_context and tier1_context != "No previous context available.":
                sections.append(f"## Recent Conversation\n{tier1_context}")
                remaining_tokens -= len(tier1_context.split()) * 2
        
        # Tier 2: Session summaries
        if include_tier2 and project_id:
            tier2_tokens = min(remaining_tokens // 2, 1000)
            tier2_context = await self.tier2.get_context_string(
                project_id, max_summaries=3
            )
            if tier2_context and tier2_context != "No previous session summaries available.":
                sections.append(f"## Previous Sessions\n{tier2_context}")
                remaining_tokens -= len(tier2_context.split()) * 2
        
        # Tier 3: Semantic memories (if query provided)
        if include_tier3 and project_id and query:
            tier3_context = await self.tier3.get_context_for_query(
                project_id, query, max_tokens=remaining_tokens
            )
            if tier3_context:
                sections.append(f"## Relevant Knowledge\n{tier3_context}")
        
        return "\n\n".join(sections) if sections else ""
    
    async def get_messages_for_llm(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get recent messages in LLM format."""
        return await self.tier1.get_messages_for_llm(session_id, limit)
    
    # Session management
    
    async def end_session(
        self,
        session_id: str,
        project_id: str,
        goal: str,
        task_id: Optional[str] = None,
        artifacts_created: Optional[List[str]] = None
    ) -> SessionSummary:
        """
        End a session, creating a Tier 2 summary.
        
        This should be called when:
        - A task completes
        - User explicitly ends session
        - Tier 1 reaches capacity
        
        Args:
            session_id: Session to summarize
            project_id: Associated project
            goal: What the session was trying to accomplish
            task_id: Associated task
            artifacts_created: Artifacts created during session
            
        Returns:
            Created session summary
        """
        # Get recent messages
        messages = await self.tier1.get_messages_for_llm(session_id)
        
        # Identify agents used
        recent = await self.tier1.get_recent_messages(session_id)
        agents_used = list(set(
            m.agent_invoked for m in recent
            if m.agent_invoked
        ))
        
        # Create Tier 2 summary
        summary = await self.tier2.create_summary(
            project_id=project_id,
            session_id=session_id,
            goal=goal,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            agents_used=agents_used,
            task_id=task_id,
            artifacts_created=artifacts_created,
        )
        
        # Extract long-term memories from the session
        full_content = "\n".join([m["content"] for m in messages])
        await self.tier3.extract_and_store(
            project_id=project_id,
            content=full_content,
            source=f"session:{session_id}",
            session_id=session_id,
        )
        
        # Clear Tier 1 for this session
        await self.tier1.clear_session(session_id)
        
        logger.info(
            "Ended session",
            session_id=session_id,
            project_id=project_id,
            summary_id=summary.id,
        )
        
        return summary
    
    # Memory operations
    
    async def store_memory(
        self,
        project_id: str,
        content: str,
        category: MemoryCategory = MemoryCategory.OTHER,
        source: str = "",
        importance: float = 0.5
    ) -> LongTermMemory:
        """Directly store a memory in Tier 3."""
        return await self.tier3.store_memory(
            project_id=project_id,
            content=content,
            category=category,
            source=source,
            importance=importance,
        )
    
    async def search_memories(
        self,
        project_id: str,
        query: str,
        limit: int = 5
    ) -> List[Tuple[LongTermMemory, float]]:
        """Search Tier 3 memories."""
        return await self.tier3.search(project_id, query, limit)
    
    async def get_decisions(
        self,
        project_id: str,
        limit: int = 10
    ) -> List[LongTermMemory]:
        """Get all decision memories."""
        return await self.tier3.get_by_category(
            project_id, MemoryCategory.DECISION, limit
        )
    
    async def get_follow_ups(
        self,
        project_id: str
    ) -> List[str]:
        """Get pending follow-up items from Tier 2."""
        return await self.tier2.get_follow_up_items(project_id)
    
    # Maintenance
    
    async def compress_if_needed(
        self,
        session_id: str,
        project_id: str,
        goal: str
    ) -> Optional[SessionSummary]:
        """
        Compress Tier 1 to Tier 2 if at capacity.
        
        Returns:
            Created summary if compression occurred
        """
        message_count = await self.tier1.get_message_count(session_id)
        
        if message_count >= self.tier1.max_messages:
            return await self.end_session(
                session_id=session_id,
                project_id=project_id,
                goal=goal,
            )
        
        return None
    
    async def decay_old_memories(
        self,
        project_id: str,
        days_threshold: int = 30
    ) -> int:
        """Apply decay to old Tier 3 memories."""
        return await self.tier3.decay_old_memories(project_id, days_threshold)
    
    # Private methods
    
    async def _maybe_extract_memories(
        self,
        project_id: str,
        content: str,
        session_id: str
    ) -> None:
        """Extract memories if content seems important."""
        # Simple heuristic: look for decision keywords
        important_keywords = [
            "decided", "important", "remember", "critical",
            "key finding", "conclusion", "recommend", "should",
            "must", "priority", "deadline", "budget"
        ]
        
        content_lower = content.lower()
        if any(kw in content_lower for kw in important_keywords):
            try:
                await self.tier3.extract_and_store(
                    project_id=project_id,
                    content=content,
                    source=f"session:{session_id}",
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"Memory extraction skipped: {e}")


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
