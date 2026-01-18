"""
Memory Data Models for Kautilya.

Defines all data structures used by the memory system.

Module: kautilya/memory/models.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import json
import uuid


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Optional metadata
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        d = {
            "role": self.role.value if isinstance(self.role, Enum) else self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(d["role"]) if isinstance(d["role"], str) else d["role"],
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d["timestamp"], str) else d["timestamp"],
            message_id=d.get("message_id", str(uuid.uuid4())),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        msg = {
            "role": self.role.value if isinstance(self.role, Enum) else self.role,
            "content": self.content,
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class SourceEntry:
    """A source used in generating a response."""
    source_type: str  # web_search, web_fetch, file_read, mcp_call, etc.
    location: str     # URL, file path, tool name
    description: str  # What was found/done
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "location": self.location,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceEntry":
        return cls(
            source_type=d["source_type"],
            location=d["location"],
            description=d["description"],
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d["timestamp"], str) else d.get("timestamp", datetime.now()),
        )


@dataclass
class Interaction:
    """A complete user-agent interaction (query + response)."""
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Content
    user_query: str = ""
    agent_response: str = ""

    # Execution details
    tools_used: List[str] = field(default_factory=list)
    sources: List[SourceEntry] = field(default_factory=list)
    iterations: int = 0

    # Metadata
    topic_tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    importance_score: float = 0.5

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_query": self.user_query,
            "agent_response": self.agent_response,
            "tools_used": self.tools_used,
            "sources": [s.to_dict() for s in self.sources],
            "iterations": self.iterations,
            "topic_tags": self.topic_tags,
            "entities": self.entities,
            "importance_score": self.importance_score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Interaction":
        return cls(
            interaction_id=d.get("interaction_id", str(uuid.uuid4())),
            session_id=d.get("session_id", ""),
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d.get("timestamp"), str) else datetime.now(),
            user_query=d.get("user_query", ""),
            agent_response=d.get("agent_response", ""),
            tools_used=d.get("tools_used", []),
            sources=[SourceEntry.from_dict(s) for s in d.get("sources", [])],
            iterations=d.get("iterations", 0),
            topic_tags=d.get("topic_tags", []),
            entities=d.get("entities", []),
            importance_score=d.get("importance_score", 0.5),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
        )


@dataclass
class Session:
    """A conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    is_active: bool = True

    # Session metadata
    title: Optional[str] = None  # Auto-generated from first query
    topic_tags: List[str] = field(default_factory=list)

    # Statistics
    message_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "is_active": self.is_active,
            "title": self.title,
            "topic_tags": self.topic_tags,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        return cls(
            session_id=d.get("session_id", str(uuid.uuid4())),
            user_id=d.get("user_id", "default"),
            started_at=datetime.fromisoformat(d["started_at"]) if isinstance(d.get("started_at"), str) else datetime.now(),
            ended_at=datetime.fromisoformat(d["ended_at"]) if d.get("ended_at") else None,
            is_active=d.get("is_active", True),
            title=d.get("title"),
            topic_tags=d.get("topic_tags", []),
            message_count=d.get("message_count", 0),
            tool_call_count=d.get("tool_call_count", 0),
            total_tokens=d.get("total_tokens", 0),
        )


@dataclass
class WorkingMemoryState:
    """Current state of working memory for a session."""
    session_id: str
    messages: List[Message] = field(default_factory=list)

    # Current context tracking
    active_topics: List[str] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)
    attention_focus: Optional[str] = None  # Main topic user is focused on

    # Recent interactions summary
    recent_tools: List[str] = field(default_factory=list)
    recent_sources: List[str] = field(default_factory=list)

    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "active_topics": self.active_topics,
            "active_entities": self.active_entities,
            "attention_focus": self.attention_focus,
            "recent_tools": self.recent_tools,
            "recent_sources": self.recent_sources,
            "last_updated": self.last_updated.isoformat(),
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkingMemoryState":
        return cls(
            session_id=d["session_id"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            active_topics=d.get("active_topics", []),
            active_entities=d.get("active_entities", []),
            attention_focus=d.get("attention_focus"),
            recent_tools=d.get("recent_tools", []),
            recent_sources=d.get("recent_sources", []),
            last_updated=datetime.fromisoformat(d["last_updated"]) if isinstance(d.get("last_updated"), str) else datetime.now(),
            message_count=d.get("message_count", 0),
        )


@dataclass
class UserProfile:
    """Aggregated user profile from semantic memories."""
    user_id: str = "default"

    # Communication preferences
    preferred_detail_level: Literal["brief", "moderate", "detailed"] = "moderate"
    preferred_format: Literal["prose", "bullets", "technical"] = "prose"
    tone_preference: Literal["formal", "casual", "professional"] = "professional"

    # Expertise (domain -> confidence 0.0-1.0)
    expertise_areas: Dict[str, float] = field(default_factory=dict)

    # Interests
    topics_of_interest: List[str] = field(default_factory=list)
    recurring_queries: List[str] = field(default_factory=list)

    # Context
    inferred_role: Optional[str] = None
    inferred_industry: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferred_detail_level": self.preferred_detail_level,
            "preferred_format": self.preferred_format,
            "tone_preference": self.tone_preference,
            "expertise_areas": self.expertise_areas,
            "topics_of_interest": self.topics_of_interest,
            "recurring_queries": self.recurring_queries,
            "inferred_role": self.inferred_role,
            "inferred_industry": self.inferred_industry,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=d.get("user_id", "default"),
            preferred_detail_level=d.get("preferred_detail_level", "moderate"),
            preferred_format=d.get("preferred_format", "prose"),
            tone_preference=d.get("tone_preference", "professional"),
            expertise_areas=d.get("expertise_areas", {}),
            topics_of_interest=d.get("topics_of_interest", []),
            recurring_queries=d.get("recurring_queries", []),
            inferred_role=d.get("inferred_role"),
            inferred_industry=d.get("inferred_industry"),
            created_at=datetime.fromisoformat(d["created_at"]) if isinstance(d.get("created_at"), str) else datetime.now(),
            last_updated=datetime.fromisoformat(d["last_updated"]) if isinstance(d.get("last_updated"), str) else datetime.now(),
        )

    def to_prompt(self) -> str:
        """Format profile for inclusion in system prompt."""
        lines = []

        if self.preferred_detail_level != "moderate":
            lines.append(f"- User prefers {self.preferred_detail_level} responses")

        if self.preferred_format != "prose":
            lines.append(f"- User prefers {self.preferred_format} format")

        if self.expertise_areas:
            high_expertise = [k for k, v in self.expertise_areas.items() if v > 0.7]
            if high_expertise:
                lines.append(f"- User has expertise in: {', '.join(high_expertise)}")

        if self.topics_of_interest:
            lines.append(f"- User is interested in: {', '.join(self.topics_of_interest[:5])}")

        if self.inferred_role:
            lines.append(f"- User appears to be a {self.inferred_role}")

        if self.inferred_industry:
            lines.append(f"- User works in {self.inferred_industry}")

        return "\n".join(lines) if lines else "No specific preferences known."


@dataclass
class MemoryContext:
    """
    Context retrieved from memory for a query.
    Passed to LLM to enhance response generation.
    """
    # Working memory (current session)
    working_messages: List[Message] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)

    # Episodic memory (relevant past interactions) - Phase 2
    relevant_episodes: List[Interaction] = field(default_factory=list)

    # Semantic memory (facts and preferences) - Phase 3
    user_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Procedural memory (applicable patterns) - Phase 4
    applicable_procedures: List[Dict[str, Any]] = field(default_factory=list)

    # User profile
    user_profile: Optional[UserProfile] = None

    def to_prompt_context(self) -> str:
        """Format memory context for inclusion in prompt."""
        sections = []

        # User profile
        if self.user_profile:
            profile_text = self.user_profile.to_prompt()
            if profile_text and profile_text != "No specific preferences known.":
                sections.append(f"## User Profile\n{profile_text}")

        # Relevant past conversations (Phase 2)
        if self.relevant_episodes:
            episodes_text = []
            for ep in self.relevant_episodes[:3]:
                query_preview = ep.user_query[:100] + "..." if len(ep.user_query) > 100 else ep.user_query
                response_preview = ep.agent_response[:150] + "..." if len(ep.agent_response) > 150 else ep.agent_response
                episodes_text.append(f"- **Q**: {query_preview}\n  **A**: {response_preview}")
            if episodes_text:
                sections.append(f"## Relevant Past Conversations\n" + "\n".join(episodes_text))

        # Known facts (Phase 3)
        if self.user_facts:
            facts_text = [f"- {f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}" for f in self.user_facts[:5]]
            sections.append(f"## Known Facts\n" + "\n".join(facts_text))

        return "\n\n".join(sections) if sections else ""


@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    # User identification
    user_id: str = "default"

    # Storage settings
    storage_backend: Literal["redis", "sqlite", "memory"] = "sqlite"
    redis_url: Optional[str] = None
    sqlite_path: Optional[str] = None

    # Working memory settings
    max_working_messages: int = 50
    working_memory_ttl: int = 3600  # 1 hour

    # Retrieval settings
    retrieval_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.5,
        "recency": 0.3,
        "importance": 0.2,
    })
    episodic_limit: int = 5
    semantic_limit: int = 10

    # Consolidation settings
    consolidation_enabled: bool = True
    consolidation_interval: str = "daily"
    min_episodes_for_consolidation: int = 5

    # Privacy settings
    local_only: bool = True
    anonymize_pii: bool = False
    retention_days: int = 90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "storage_backend": self.storage_backend,
            "redis_url": self.redis_url,
            "sqlite_path": self.sqlite_path,
            "max_working_messages": self.max_working_messages,
            "working_memory_ttl": self.working_memory_ttl,
            "retrieval_weights": self.retrieval_weights,
            "episodic_limit": self.episodic_limit,
            "semantic_limit": self.semantic_limit,
            "consolidation_enabled": self.consolidation_enabled,
            "consolidation_interval": self.consolidation_interval,
            "min_episodes_for_consolidation": self.min_episodes_for_consolidation,
            "local_only": self.local_only,
            "anonymize_pii": self.anonymize_pii,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
