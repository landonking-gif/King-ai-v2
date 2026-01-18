"""
Memory Manager for Kautilya.

Central coordinator for all memory types:
- Working Memory: Current session context
- Episodic Memory: Past interactions (Phase 2)
- Semantic Memory: Learned facts (Phase 3)
- Procedural Memory: Successful patterns (Phase 4)

Module: kautilya/memory/manager.py
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    MemoryConfig,
    MemoryContext,
    Message,
    MessageRole,
    Interaction,
    Session,
    UserProfile,
    SourceEntry,
)
from .storage import StorageBackend, get_storage, SQLiteStorage
from .working import WorkingMemoryStore
from .session import SessionManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory manager for Kautilya.

    Coordinates all memory types and provides unified interface for:
    - Remembering interactions
    - Recalling relevant context
    - Managing sessions
    - Tracking user preferences
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        storage: Optional[StorageBackend] = None,
    ):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
            storage: Optional storage backend (auto-created if not provided)
        """
        self.config = config or MemoryConfig()

        # Initialize storage
        if storage:
            self.storage = storage
        else:
            self.storage = get_storage(
                backend=self.config.storage_backend,
                redis_url=self.config.redis_url or os.getenv("REDIS_URL"),
                sqlite_path=self.config.sqlite_path,
            )

        # Initialize memory stores
        self.working = WorkingMemoryStore(
            storage=self.storage,
            max_messages=self.config.max_working_messages,
            ttl=self.config.working_memory_ttl,
        )

        # Initialize session manager
        self.sessions = SessionManager(
            storage=self.storage,
            user_id=self.config.user_id,
        )

        # Current session tracking
        self._current_session: Optional[Session] = None

        logger.info(f"Initialized MemoryManager with {type(self.storage).__name__}")

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        session = self.get_or_create_session()
        return session.session_id

    @property
    def user_id(self) -> str:
        """Get current user ID."""
        return self.sessions.user_id

    # === Session Management ===

    def get_or_create_session(self) -> Session:
        """
        Get current session or create new one.

        Returns:
            Active Session
        """
        if self._current_session:
            return self._current_session

        self._current_session = self.sessions.get_or_create_session()
        return self._current_session

    def create_session(self, title: Optional[str] = None) -> Session:
        """
        Create a new session.

        Args:
            title: Optional session title

        Returns:
            New Session
        """
        self._current_session = self.sessions.create_session(title=title)
        return self._current_session

    def end_session(self) -> bool:
        """
        End the current session.

        Returns:
            Success boolean
        """
        if self._current_session:
            result = self.sessions.end_session(self._current_session.session_id)
            self._current_session = None
            return result
        return False

    def get_recent_sessions(self, limit: int = 10) -> List[Session]:
        """Get recent sessions."""
        return self.sessions.get_recent_sessions(limit)

    def resume_session(self, session_id: str) -> Optional[Session]:
        """
        Resume a previous session.

        Args:
            session_id: Session ID to resume

        Returns:
            Resumed Session or None
        """
        session = self.sessions.resume_session(session_id)
        if session:
            self._current_session = session
        return session

    # === Memory Operations ===

    def remember(
        self,
        user_query: str,
        agent_response: str,
        tools_used: Optional[List[str]] = None,
        sources: Optional[List[SourceEntry]] = None,
        iterations: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Interaction:
        """
        Remember an interaction.

        Stores the interaction in:
        - Working memory (messages)
        - Episodic memory (full interaction record)

        Args:
            user_query: User's query
            agent_response: Agent's response
            tools_used: List of tools used
            sources: List of sources consulted
            iterations: Number of iterations
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Created Interaction
        """
        session = self.get_or_create_session()

        # Add to working memory
        self.working.add_user_message(session.session_id, user_query)
        self.working.add_assistant_message(session.session_id, agent_response)

        # Track tools
        for tool in (tools_used or []):
            self.working.add_tool_usage(session.session_id, tool)

        # Track sources
        for source in (sources or []):
            self.working.add_source(session.session_id, source.location)

        # Create interaction record
        interaction = Interaction(
            session_id=session.session_id,
            user_query=user_query,
            agent_response=agent_response,
            tools_used=tools_used or [],
            sources=sources or [],
            iterations=iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Extract topics and entities
        context = self.working.get_state(session.session_id)
        interaction.topic_tags = context.active_topics[:5]
        interaction.entities = context.active_entities[:10]

        # Save interaction
        self.sessions.save_interaction(interaction)

        # Update session stats
        self.sessions.update_session_stats(
            session.session_id,
            message_count_delta=2,  # user + assistant
            tool_call_count_delta=len(tools_used or []),
            tokens_delta=input_tokens + output_tokens,
        )

        # Auto-generate session title from first query
        if session.message_count == 0 and not session.title:
            title = self.sessions.generate_session_title(user_query)
            self.sessions.update_session_title(session.session_id, title)

        # Add topics to session
        for topic in interaction.topic_tags:
            self.sessions.add_session_topic(session.session_id, topic)

        logger.debug(f"Remembered interaction: {interaction.interaction_id}")

        return interaction

    def recall(
        self,
        query: str,
        include_working: bool = True,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_profile: bool = True,
    ) -> MemoryContext:
        """
        Recall relevant context for a query.

        Retrieves:
        - Working memory (current session)
        - Episodic memory (relevant past interactions)
        - Semantic memory (facts and preferences)
        - User profile

        Args:
            query: Query to find context for
            include_working: Include working memory
            include_episodic: Include episodic memory
            include_semantic: Include semantic memory
            include_profile: Include user profile

        Returns:
            MemoryContext with relevant memories
        """
        session = self.get_or_create_session()
        context = MemoryContext()

        # Get working memory
        if include_working:
            state = self.working.get_state(session.session_id)
            context.working_messages = state.messages
            context.active_topics = state.active_topics

        # Get episodic memory (Phase 2 - for now just recent interactions)
        if include_episodic:
            recent = self.sessions.get_recent_interactions(
                limit=self.config.episodic_limit
            )
            context.relevant_episodes = recent

        # Get semantic memory (Phase 3 - placeholder)
        if include_semantic:
            # TODO: Implement semantic memory retrieval
            context.user_facts = []

        # Get user profile
        if include_profile:
            context.user_profile = self.sessions.get_user_profile()

        return context

    def get_messages_for_llm(
        self,
        system_prompt: Optional[str] = None,
        include_memory_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get messages formatted for LLM API.

        Args:
            system_prompt: Base system prompt
            include_memory_context: Whether to enhance with memory context

        Returns:
            List of message dicts for LLM
        """
        session = self.get_or_create_session()

        # Build enhanced system prompt
        if include_memory_context:
            context = self.recall("")
            memory_section = context.to_prompt_context()

            if memory_section:
                system_prompt = (system_prompt or "") + "\n\n" + memory_section

        return self.working.get_messages_for_llm(
            session.session_id,
            system_prompt=system_prompt,
        )

    def add_message(
        self,
        role: MessageRole,
        content: str,
        **kwargs,
    ) -> Message:
        """
        Add a single message to working memory.

        Args:
            role: Message role
            content: Message content
            **kwargs: Additional message fields

        Returns:
            Created Message
        """
        session = self.get_or_create_session()
        return self.working.add_message(
            session.session_id,
            role,
            content,
            **kwargs,
        )

    def add_user_message(self, content: str) -> Message:
        """Add user message."""
        return self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """Add assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def add_tool_message(self, content: str, tool_call_id: str, name: str) -> Message:
        """Add tool message."""
        session = self.get_or_create_session()
        return self.working.add_tool_message(
            session.session_id,
            content,
            tool_call_id,
            name,
        )

    # === User Profile ===

    def get_user_profile(self) -> UserProfile:
        """Get user profile."""
        return self.sessions.get_user_profile()

    def update_user_profile(self, updates: Dict[str, Any]) -> UserProfile:
        """Update user profile."""
        return self.sessions.update_user_profile(updates)

    # === Context Helpers ===

    def get_recent_context(self) -> Dict[str, Any]:
        """
        Get summary of recent context.

        Returns:
            Dict with topics, entities, recent queries
        """
        session = self.get_or_create_session()
        return self.working.get_recent_context(session.session_id)

    def get_attention_focus(self) -> Optional[str]:
        """Get current attention focus."""
        session = self.get_or_create_session()
        state = self.working.get_state(session.session_id)
        return state.attention_focus

    def set_attention_focus(self, focus: str) -> None:
        """Set current attention focus."""
        session = self.get_or_create_session()
        self.working.update_attention_focus(session.session_id, focus)

    # === Memory Management ===

    def clear_working_memory(self) -> bool:
        """Clear working memory for current session."""
        session = self.get_or_create_session()
        return self.working.clear(session.session_id)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dict with memory stats
        """
        session = self.get_or_create_session()
        state = self.working.get_state(session.session_id)
        recent_sessions = self.sessions.get_recent_sessions(limit=100)

        return {
            "user_id": self.user_id,
            "current_session_id": session.session_id,
            "working_memory": {
                "message_count": state.message_count,
                "active_topics": len(state.active_topics),
                "active_entities": len(state.active_entities),
                "recent_tools": len(state.recent_tools),
            },
            "sessions": {
                "total_sessions": len(recent_sessions),
                "active_session": session.is_active,
                "session_started": session.started_at.isoformat(),
            },
            "storage_backend": type(self.storage).__name__,
        }

    def search_history(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Interaction]:
        """
        Search interaction history.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching Interactions
        """
        # Simple keyword search for Phase 1
        # Phase 2 will add vector search
        interactions = self.sessions.get_recent_interactions(limit=100)

        query_lower = query.lower()
        matches = []

        for interaction in interactions:
            if (query_lower in interaction.user_query.lower() or
                query_lower in interaction.agent_response.lower()):
                matches.append(interaction)

                if len(matches) >= limit:
                    break

        return matches

    def close(self) -> None:
        """Close memory manager and release resources."""
        if hasattr(self.storage, 'close'):
            self.storage.close()
        logger.info("MemoryManager closed")


# === Global Memory Manager Instance ===

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    config: Optional[MemoryConfig] = None,
    reinitialize: bool = False,
) -> MemoryManager:
    """
    Get global MemoryManager instance.

    Args:
        config: Optional configuration (used only on first call or reinitialize)
        reinitialize: Force re-initialization

    Returns:
        MemoryManager instance
    """
    global _memory_manager

    if _memory_manager is None or reinitialize:
        _memory_manager = MemoryManager(config=config)

    return _memory_manager


def clear_memory_manager() -> None:
    """Clear global MemoryManager instance."""
    global _memory_manager

    if _memory_manager:
        _memory_manager.close()
        _memory_manager = None
