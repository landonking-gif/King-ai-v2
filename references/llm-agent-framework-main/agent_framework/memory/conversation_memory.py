"""
Conversation Memory for Agent Framework

Memory systems for maintaining context across interactions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    """A single message in conversation"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class ConversationMemory:
    """
    Memory for storing conversation history.
    
    Supports:
    - Full conversation history
    - Sliding window memory
    - Summary-based memory
    """
    
    def __init__(self, max_messages: Optional[int] = None):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.summaries: List[str] = []
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to memory"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        
        # Trim if exceeds max
        if self.max_messages and len(self.messages) > self.max_messages:
            # Keep system messages and trim oldest
            system_msgs = [m for m in self.messages if m.role == 'system']
            other_msgs = [m for m in self.messages if m.role != 'system']
            
            # Keep most recent messages
            keep_count = self.max_messages - len(system_msgs)
            self.messages = system_msgs + other_msgs[-keep_count:]
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory"""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def get_context_string(self, limit: Optional[int] = None) -> str:
        """Get conversation as formatted string"""
        messages = self.get_messages(limit)
        
        formatted = []
        for msg in messages:
            role_label = msg.role.capitalize()
            formatted.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(formatted)
    
    def get_messages_for_llm(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages in LLM-compatible format"""
        messages = self.get_messages(limit)
        return [{'role': m.role, 'content': m.content} for m in messages]
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages = []
    
    def search(self, query: str, limit: int = 5) -> List[Message]:
        """Search messages containing query"""
        results = []
        query_lower = query.lower()
        
        for msg in reversed(self.messages):
            if query_lower in msg.content.lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save memory to file"""
        data = {
            'messages': [m.to_dict() for m in self.messages],
            'summaries': self.summaries
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load memory from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.messages = []
        for m in data.get('messages', []):
            self.messages.append(Message(
                role=m['role'],
                content=m['content'],
                timestamp=datetime.fromisoformat(m['timestamp']),
                metadata=m.get('metadata', {})
            ))
        
        self.summaries = data.get('summaries', [])


class SummaryMemory(ConversationMemory):
    """
    Memory that summarizes old conversations.
    
    When conversation exceeds threshold, older messages
    are summarized to save context space.
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        summary_threshold: int = 15,
        summarizer: Any = None
    ):
        super().__init__(max_messages)
        self.summary_threshold = summary_threshold
        self.summarizer = summarizer
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add message and summarize if needed"""
        super().add_message(role, content, metadata)
        
        # Check if we need to summarize
        if len(self.messages) > self.summary_threshold:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self) -> None:
        """Summarize older messages"""
        if not self.summarizer:
            # Simple truncation if no summarizer
            keep_count = self.summary_threshold // 2
            self.messages = self.messages[-keep_count:]
            return
        
        # Get messages to summarize
        to_summarize = self.messages[:-self.summary_threshold // 2]
        
        # Create summary
        context = "\n".join([f"{m.role}: {m.content}" for m in to_summarize])
        summary = self.summarizer.summarize(context)
        
        # Store summary
        self.summaries.append(summary)
        
        # Keep recent messages
        self.messages = self.messages[-self.summary_threshold // 2:]
    
    def get_full_context(self) -> str:
        """Get full context including summaries"""
        parts = []
        
        if self.summaries:
            parts.append("Previous conversation summary:")
            parts.extend(self.summaries)
            parts.append("\nRecent conversation:")
        
        parts.append(self.get_context_string())
        
        return "\n".join(parts)


class EntityMemory:
    """
    Memory for tracking entities mentioned in conversation.
    
    Useful for maintaining context about people, places,
    and things discussed.
    """
    
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Dict[str, Any] = None
    ) -> None:
        """Add or update an entity"""
        if name not in self.entities:
            self.entities[name] = {
                'type': entity_type,
                'attributes': attributes or {},
                'mentions': 1,
                'first_seen': datetime.now(),
                'last_seen': datetime.now()
            }
        else:
            self.entities[name]['mentions'] += 1
            self.entities[name]['last_seen'] = datetime.now()
            if attributes:
                self.entities[name]['attributes'].update(attributes)
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name"""
        return self.entities.get(name)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a type"""
        return [
            {'name': name, **data}
            for name, data in self.entities.items()
            if data['type'] == entity_type
        ]
    
    def get_context_string(self) -> str:
        """Get entities as context string"""
        if not self.entities:
            return ""
        
        lines = ["Known entities:"]
        for name, data in self.entities.items():
            attrs = ", ".join([f"{k}={v}" for k, v in data['attributes'].items()])
            lines.append(f"- {name} ({data['type']}): {attrs}")
        
        return "\n".join(lines)
