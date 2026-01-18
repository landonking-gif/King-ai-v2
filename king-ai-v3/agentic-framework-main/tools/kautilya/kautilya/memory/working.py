"""
Working Memory Store for Kautilya.

Manages current session context with:
- Message history (with smart trimming)
- Active topic tracking
- Entity recognition
- Attention focus management

Module: kautilya/memory/working.py
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    Message,
    MessageRole,
    WorkingMemoryState,
    Interaction,
)
from .storage import StorageBackend

logger = logging.getLogger(__name__)


class WorkingMemoryStore:
    """
    Manages working memory for the current conversation session.

    Working memory holds:
    - Recent messages in the conversation
    - Active topics being discussed
    - Entities mentioned
    - Current attention focus
    """

    # Key prefixes for storage
    KEY_PREFIX = "kautilya:working:"

    def __init__(
        self,
        storage: StorageBackend,
        max_messages: int = 50,
        ttl: int = 3600,  # 1 hour default
    ):
        """
        Initialize working memory store.

        Args:
            storage: Storage backend instance
            max_messages: Maximum messages to keep in working memory
            ttl: Time-to-live for working memory in seconds
        """
        self.storage = storage
        self.max_messages = max_messages
        self.ttl = ttl

    def _get_key(self, session_id: str) -> str:
        """Get storage key for session."""
        return f"{self.KEY_PREFIX}{session_id}"

    def get_state(self, session_id: str) -> WorkingMemoryState:
        """
        Get working memory state for session.

        Args:
            session_id: Session identifier

        Returns:
            WorkingMemoryState (empty if not found)
        """
        key = self._get_key(session_id)
        data = self.storage.get_json(key)

        if data:
            return WorkingMemoryState.from_dict(data)

        # Return empty state for new session
        return WorkingMemoryState(session_id=session_id)

    def save_state(self, state: WorkingMemoryState) -> bool:
        """
        Save working memory state.

        Args:
            state: WorkingMemoryState to save

        Returns:
            Success boolean
        """
        state.last_updated = datetime.now()
        state.message_count = len(state.messages)

        key = self._get_key(state.session_id)
        return self.storage.set_json(key, state.to_dict(), ttl=self.ttl)

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Message:
        """
        Add a message to working memory.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, tool, system)
            content: Message content
            tool_calls: Optional tool calls (for assistant messages)
            tool_call_id: Optional tool call ID (for tool messages)
            name: Optional name (for tool messages)

        Returns:
            The created Message
        """
        state = self.get_state(session_id)

        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        )

        state.messages.append(message)

        # Trim if exceeding max
        if len(state.messages) > self.max_messages:
            state.messages = self._trim_messages(state.messages)

        # Update context tracking
        if role == MessageRole.USER:
            self._update_context_from_user_message(state, content)
        elif role == MessageRole.ASSISTANT:
            self._update_context_from_assistant_message(state, content)

        self.save_state(state)
        return message

    def add_user_message(self, session_id: str, content: str) -> Message:
        """Add a user message."""
        return self.add_message(session_id, MessageRole.USER, content)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Add an assistant message."""
        return self.add_message(
            session_id,
            MessageRole.ASSISTANT,
            content,
            tool_calls=tool_calls,
        )

    def add_tool_message(
        self,
        session_id: str,
        content: str,
        tool_call_id: str,
        name: str,
    ) -> Message:
        """Add a tool result message."""
        return self.add_message(
            session_id,
            MessageRole.TOOL,
            content,
            tool_call_id=tool_call_id,
            name=name,
        )

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Message]:
        """
        Get messages from working memory.

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            include_system: Whether to include system messages

        Returns:
            List of Messages
        """
        state = self.get_state(session_id)
        messages = state.messages

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_messages_for_llm(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get messages formatted for LLM API.

        Args:
            session_id: Session identifier
            system_prompt: Optional system prompt to prepend

        Returns:
            List of message dicts in OpenAI format
        """
        messages = self.get_messages(session_id, include_system=False)

        result = []

        # Add system prompt if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # Convert messages to OpenAI format
        for msg in messages:
            result.append(msg.to_openai_format())

        return result

    def get_recent_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get recent context summary for retrieval augmentation.

        Args:
            session_id: Session identifier

        Returns:
            Context dict with topics, entities, recent queries
        """
        state = self.get_state(session_id)

        # Get recent user queries
        recent_queries = []
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                recent_queries.append(msg.content[:100])
                if len(recent_queries) >= 3:
                    break

        return {
            "active_topics": state.active_topics,
            "active_entities": state.active_entities,
            "attention_focus": state.attention_focus,
            "recent_queries": recent_queries,
            "recent_tools": state.recent_tools,
            "message_count": state.message_count,
        }

    def update_attention_focus(self, session_id: str, focus: str) -> None:
        """
        Update the current attention focus.

        Args:
            session_id: Session identifier
            focus: Main topic user is focused on
        """
        state = self.get_state(session_id)
        state.attention_focus = focus
        self.save_state(state)

    def add_tool_usage(self, session_id: str, tool_name: str) -> None:
        """
        Track tool usage in working memory.

        Args:
            session_id: Session identifier
            tool_name: Name of tool used
        """
        state = self.get_state(session_id)

        if tool_name not in state.recent_tools:
            state.recent_tools.append(tool_name)

        # Keep only recent tools
        state.recent_tools = state.recent_tools[-10:]

        self.save_state(state)

    def add_source(self, session_id: str, source: str) -> None:
        """
        Track source consulted in working memory.

        Args:
            session_id: Session identifier
            source: Source description
        """
        state = self.get_state(session_id)

        if source not in state.recent_sources:
            state.recent_sources.append(source)

        # Keep only recent sources
        state.recent_sources = state.recent_sources[-10:]

        self.save_state(state)

    def clear(self, session_id: str) -> bool:
        """
        Clear working memory for session.

        Args:
            session_id: Session identifier

        Returns:
            Success boolean
        """
        key = self._get_key(session_id)
        return self.storage.delete(key)

    def _trim_messages(self, messages: List[Message]) -> List[Message]:
        """
        Trim messages while preserving tool call pairs.

        Ensures tool calls and their responses stay together.

        Args:
            messages: List of messages to trim

        Returns:
            Trimmed list of messages
        """
        if len(messages) <= self.max_messages:
            return messages

        # Always keep system messages
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in messages if m.role != MessageRole.SYSTEM]

        # Calculate how many non-system messages to keep
        keep_count = self.max_messages - len(system_msgs)

        if keep_count <= 0:
            return system_msgs[:self.max_messages]

        # Find safe trim point (don't split tool call pairs)
        trimmed = other_msgs[-keep_count:]

        # If first message is a tool response, include preceding tool call
        if trimmed and trimmed[0].role == MessageRole.TOOL:
            # Find the tool call this responds to
            tool_call_id = trimmed[0].tool_call_id
            if tool_call_id:
                for i, msg in enumerate(other_msgs):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.get("id") == tool_call_id:
                                # Include this message and all after
                                trimmed = other_msgs[i:][-keep_count:]
                                break

        return system_msgs + trimmed

    def _update_context_from_user_message(
        self,
        state: WorkingMemoryState,
        content: str,
    ) -> None:
        """Extract and update context from user message."""
        # Extract entities (simple capitalized word extraction)
        entities = self._extract_entities(content)
        for entity in entities:
            if entity not in state.active_entities:
                state.active_entities.append(entity)

        # Keep only recent entities
        state.active_entities = state.active_entities[-20:]

        # Extract topics (simple keyword extraction)
        topics = self._extract_topics(content)
        for topic in topics:
            if topic not in state.active_topics:
                state.active_topics.append(topic)

        # Keep only recent topics
        state.active_topics = state.active_topics[-10:]

        # Update attention focus to most recent significant topic
        if topics:
            state.attention_focus = topics[0]

    def _update_context_from_assistant_message(
        self,
        state: WorkingMemoryState,
        content: str,
    ) -> None:
        """Extract and update context from assistant message."""
        # Extract entities mentioned in response
        entities = self._extract_entities(content)
        for entity in entities[:5]:  # Limit extraction from long responses
            if entity not in state.active_entities:
                state.active_entities.append(entity)

        state.active_entities = state.active_entities[-20:]

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text (simple approach).

        For production, consider using spaCy or similar NER.
        """
        # Simple approach: find capitalized words/phrases
        # Exclude common sentence starters
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where',
            'Why', 'How', 'Who', 'Which', 'Can', 'Could', 'Would', 'Should',
            'Will', 'May', 'Might', 'Must', 'Is', 'Are', 'Was', 'Were', 'Be',
            'Been', 'Being', 'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'I',
            'You', 'He', 'She', 'It', 'We', 'They', 'My', 'Your', 'His', 'Her',
            'Its', 'Our', 'Their', 'A', 'An', 'And', 'But', 'Or', 'So', 'Yet',
            'For', 'Nor', 'As', 'If', 'Then', 'Also', 'Just', 'Now', 'Please',
            'Yes', 'No', 'Here', 'There', 'Hello', 'Hi', 'Thanks', 'Thank',
        }

        # Find capitalized words (potential entities)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)

        # Filter out common words
        entities = [m for m in matches if m not in common_words]

        return entities[:10]  # Limit to 10

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text (simple keyword approach).

        For production, consider using topic modeling or LLM extraction.
        """
        # Domain-specific topic keywords
        topic_keywords = {
            # Finance
            'market': 'market analysis',
            'stock': 'stocks',
            'share': 'stocks',
            'gold': 'precious metals',
            'silver': 'precious metals',
            'investment': 'investment',
            'trading': 'trading',
            'portfolio': 'portfolio management',
            'crypto': 'cryptocurrency',
            'bitcoin': 'cryptocurrency',

            # Tech
            'code': 'programming',
            'python': 'python',
            'javascript': 'javascript',
            'api': 'api development',
            'database': 'databases',
            'cloud': 'cloud computing',
            'ai': 'artificial intelligence',
            'machine learning': 'machine learning',

            # General
            'news': 'news',
            'weather': 'weather',
            'travel': 'travel',
            'health': 'health',
            'sports': 'sports',
        }

        text_lower = text.lower()
        topics = []

        for keyword, topic in topic_keywords.items():
            if keyword in text_lower and topic not in topics:
                topics.append(topic)

        return topics[:5]  # Limit to 5 topics


def create_working_memory_store(
    storage: StorageBackend,
    config: Optional[Dict[str, Any]] = None,
) -> WorkingMemoryStore:
    """
    Factory function to create WorkingMemoryStore.

    Args:
        storage: Storage backend instance
        config: Optional configuration dict

    Returns:
        WorkingMemoryStore instance
    """
    config = config or {}
    return WorkingMemoryStore(
        storage=storage,
        max_messages=config.get("max_working_messages", 50),
        ttl=config.get("working_memory_ttl", 3600),
    )
