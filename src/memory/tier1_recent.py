"""
Tier 1 Memory - Recent Context.

Manages the last N messages per session/project with verbatim storage.
This is the immediate context window for agent conversations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

from src.utils.structured_logging import get_logger

logger = get_logger("tier1_memory")


@dataclass
class RecentMessage:
    """A single message in recent context."""
    id: str
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_invoked: Optional[str] = None
    tool_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "agent_invoked": self.agent_invoked,
            "tool_used": self.tool_used,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecentMessage":
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def to_llm_format(self) -> Dict[str, str]:
        """Convert to LLM-compatible format."""
        return {"role": self.role, "content": self.content}


class Tier1Memory:
    """
    Tier 1 Memory - Recent Context Window.
    
    Stores the last N messages verbatim for each session/project.
    This provides immediate context for ongoing conversations.
    
    Features:
    - Fixed-size sliding window
    - System messages preserved
    - Fast in-memory access
    - Persistence to Redis/file
    """
    
    MAX_RECENT_MESSAGES = 10
    
    def __init__(
        self,
        redis_client=None,
        storage_path: Optional[Path] = None,
        max_messages: int = 10
    ):
        """
        Initialize Tier 1 memory.
        
        Args:
            redis_client: Redis client for persistence (optional)
            storage_path: File-based storage path (fallback)
            max_messages: Maximum messages to keep per session
        """
        self.redis = redis_client
        self.storage_path = storage_path or Path("data/memory/tier1")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_messages = max_messages
        
        # In-memory cache per session
        self._cache: Dict[str, List[RecentMessage]] = {}
    
    async def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecentMessage:
        """Add a user message to the session context."""
        message = RecentMessage(
            id=f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            role="user",
            content=content,
            metadata=metadata or {},
        )
        await self._add_message(session_id, message)
        return message
    
    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        agent_invoked: Optional[str] = None,
        tool_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecentMessage:
        """Add an assistant message to the session context."""
        message = RecentMessage(
            id=f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            role="assistant",
            content=content,
            agent_invoked=agent_invoked,
            tool_used=tool_used,
            metadata=metadata or {},
        )
        await self._add_message(session_id, message)
        return message
    
    async def add_system_message(
        self,
        session_id: str,
        content: str
    ) -> RecentMessage:
        """Add a system message (these are preserved during trimming)."""
        message = RecentMessage(
            id=f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            role="system",
            content=content,
        )
        await self._add_message(session_id, message)
        return message
    
    async def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[RecentMessage]:
        """
        Get recent messages for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return (defaults to max_messages)
            
        Returns:
            List of recent messages in chronological order
        """
        messages = await self._load_messages(session_id)
        limit = limit or self.max_messages
        return messages[-limit:]
    
    async def get_context_string(
        self,
        session_id: str,
        include_roles: bool = True
    ) -> str:
        """
        Get recent context as formatted string for agent prompts.
        
        Args:
            session_id: Session identifier
            include_roles: Include role labels in output
            
        Returns:
            Formatted context string
        """
        messages = await self.get_recent_messages(session_id)
        
        if not messages:
            return "No previous context available."
        
        lines = []
        for msg in messages:
            if include_roles:
                role = msg.role.upper()
                if msg.agent_invoked:
                    role += f" (via {msg.agent_invoked})"
                lines.append(f"[{role}]: {msg.content}")
            else:
                lines.append(msg.content)
        
        return "\n\n".join(lines)
    
    async def get_messages_for_llm(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get messages in LLM-compatible format."""
        messages = await self.get_recent_messages(session_id, limit)
        return [msg.to_llm_format() for msg in messages]
    
    async def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        self._cache.pop(session_id, None)
        
        if self.redis:
            await self.redis.delete(f"tier1:{session_id}")
        else:
            file_path = self.storage_path / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Cleared Tier 1 memory", session_id=session_id)
    
    async def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        messages = await self._load_messages(session_id)
        return len(messages)
    
    # Private methods
    
    async def _add_message(
        self,
        session_id: str,
        message: RecentMessage
    ) -> None:
        """Add a message and maintain the window size."""
        messages = await self._load_messages(session_id)
        messages.append(message)
        
        # Trim to max, preserving system messages
        if len(messages) > self.max_messages:
            system_msgs = [m for m in messages if m.role == "system"]
            other_msgs = [m for m in messages if m.role != "system"]
            
            # Keep most recent non-system messages
            keep_count = self.max_messages - len(system_msgs)
            messages = system_msgs + other_msgs[-keep_count:]
        
        # Update cache and persist
        self._cache[session_id] = messages
        await self._save_messages(session_id, messages)
    
    async def _load_messages(self, session_id: str) -> List[RecentMessage]:
        """Load messages from cache or storage."""
        if session_id in self._cache:
            return self._cache[session_id]
        
        messages = []
        
        if self.redis:
            data = await self.redis.get(f"tier1:{session_id}")
            if data:
                messages = [RecentMessage.from_dict(m) for m in json.loads(data)]
        else:
            file_path = self.storage_path / f"{session_id}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    messages = [RecentMessage.from_dict(m) for m in data]
        
        self._cache[session_id] = messages
        return messages
    
    async def _save_messages(
        self,
        session_id: str,
        messages: List[RecentMessage]
    ) -> None:
        """Persist messages to storage."""
        data = [m.to_dict() for m in messages]
        
        if self.redis:
            await self.redis.set(
                f"tier1:{session_id}",
                json.dumps(data),
                ex=86400 * 7  # 7 day expiry
            )
        else:
            file_path = self.storage_path / f"{session_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
