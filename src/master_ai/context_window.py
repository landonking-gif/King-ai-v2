"""
Context Window Manager.
Manages LLM context windows with smart truncation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re

from src.utils.structured_logging import get_logger

logger = get_logger("context_window")


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class TruncationStrategy(str, Enum):
    """Strategies for truncating context."""
    FIFO = "fifo"  # Remove oldest first
    LIFO = "lifo"  # Remove newest first
    IMPORTANCE = "importance"  # Remove least important
    SMART = "smart"  # Preserve system, recent, and important
    SUMMARIZE = "summarize"  # Summarize middle content


class ContentType(str, Enum):
    """Types of content in messages."""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    MARKDOWN = "markdown"
    DATA = "data"


@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    content_type: ContentType = ContentType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


@dataclass
class ContextStats:
    """Statistics about context usage."""
    total_tokens: int
    max_tokens: int
    messages_count: int
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    usage_percent: float
    
    @property
    def remaining_tokens(self) -> int:
        return self.max_tokens - self.total_tokens


@dataclass
class TruncationResult:
    """Result of context truncation."""
    original_tokens: int
    final_tokens: int
    messages_removed: int
    messages_summarized: int
    strategy_used: TruncationStrategy


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Simple estimation: ~4 characters per token for English.
    """
    if not text:
        return 0
    
    # Basic estimation
    char_count = len(text)
    word_count = len(text.split())
    
    # Use a combination of character and word counts
    # Average English word is ~5 chars, average token is ~4 chars
    estimate = max(char_count // 4, word_count * 1.3)
    
    return int(estimate)


def calculate_importance(message: Message) -> float:
    """Calculate importance score for a message."""
    score = 0.5
    
    # Role-based importance
    if message.role == MessageRole.SYSTEM:
        score += 0.4  # System messages are critical
    elif message.role == MessageRole.TOOL:
        score += 0.2  # Tool outputs are valuable
    
    # Content-based importance
    content_lower = message.content.lower()
    
    # Important keywords
    important_keywords = [
        "error", "critical", "important", "required",
        "must", "decision", "approval", "action",
    ]
    for keyword in important_keywords:
        if keyword in content_lower:
            score += 0.1
    
    # Code and structured data are often important
    if message.content_type in (ContentType.CODE, ContentType.JSON):
        score += 0.1
    
    # Recency boost (metadata would include this)
    if message.metadata.get("is_recent"):
        score += 0.2
    
    return min(1.0, score)


class ContextWindow:
    """
    Manages a single context window for LLM interactions.
    """
    
    def __init__(
        self,
        max_tokens: int,
        reserve_tokens: int = 500,  # Reserve for response
    ):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self._messages: List[Message] = []
        self._system_message: Optional[Message] = None
    
    @property
    def available_tokens(self) -> int:
        """Tokens available for new content."""
        used = sum(m.token_count for m in self._messages)
        if self._system_message:
            used += self._system_message.token_count
        return self.max_tokens - used - self.reserve_tokens
    
    @property
    def total_tokens(self) -> int:
        """Total tokens currently in context."""
        tokens = sum(m.token_count for m in self._messages)
        if self._system_message:
            tokens += self._system_message.token_count
        return tokens
    
    def set_system_message(self, content: str) -> None:
        """Set the system message."""
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            importance=1.0,
        )
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        importance: Optional[float] = None,
        content_type: ContentType = ContentType.TEXT,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Add a message to the context.
        
        Returns:
            True if message was added, False if truncation needed
        """
        message = Message(
            role=role,
            content=content,
            importance=importance or 0.5,
            content_type=content_type,
            metadata=metadata or {},
        )
        
        # Calculate importance if not provided
        if importance is None:
            message.importance = calculate_importance(message)
        
        # Check if fits
        if message.token_count <= self.available_tokens:
            self._messages.append(message)
            return True
        
        return False
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI-compatible format."""
        messages = []
        
        if self._system_message:
            messages.append({
                "role": "system",
                "content": self._system_message.content,
            })
        
        for msg in self._messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        return messages
    
    def get_stats(self) -> ContextStats:
        """Get context usage statistics."""
        system_tokens = self._system_message.token_count if self._system_message else 0
        user_tokens = sum(m.token_count for m in self._messages if m.role == MessageRole.USER)
        assistant_tokens = sum(m.token_count for m in self._messages if m.role == MessageRole.ASSISTANT)
        total = self.total_tokens
        
        return ContextStats(
            total_tokens=total,
            max_tokens=self.max_tokens,
            messages_count=len(self._messages) + (1 if self._system_message else 0),
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            usage_percent=total / self.max_tokens * 100,
        )
    
    def clear(self) -> None:
        """Clear all messages except system."""
        self._messages.clear()


class ContextWindowManager:
    """
    Manages LLM context windows with smart truncation.
    
    Features:
    - Multiple truncation strategies
    - Importance-based retention
    - Token estimation
    - Context statistics
    """
    
    # Model context limits
    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "gemini-pro": 32000,
        "llama-2-70b": 4096,
        "mistral-7b": 8192,
    }
    
    def __init__(
        self,
        default_max_tokens: int = 4096,
        default_strategy: TruncationStrategy = TruncationStrategy.SMART,
    ):
        self.default_max_tokens = default_max_tokens
        self.default_strategy = default_strategy
        self._contexts: Dict[str, ContextWindow] = {}
    
    def get_model_limit(self, model: str) -> int:
        """Get context limit for a model."""
        return self.MODEL_LIMITS.get(model, self.default_max_tokens)
    
    def create_context(
        self,
        context_id: str,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> ContextWindow:
        """Create a new context window."""
        if model:
            max_tokens = self.get_model_limit(model)
        
        context = ContextWindow(
            max_tokens=max_tokens or self.default_max_tokens
        )
        
        self._contexts[context_id] = context
        return context
    
    def get_context(self, context_id: str) -> Optional[ContextWindow]:
        """Get an existing context."""
        return self._contexts.get(context_id)
    
    def truncate(
        self,
        context: ContextWindow,
        target_tokens: Optional[int] = None,
        strategy: Optional[TruncationStrategy] = None,
    ) -> TruncationResult:
        """
        Truncate context to fit within limits.
        
        Args:
            context: Context window to truncate
            target_tokens: Target token count (default: max - reserve)
            strategy: Truncation strategy to use
            
        Returns:
            Truncation result
        """
        strategy = strategy or self.default_strategy
        target = target_tokens or (context.max_tokens - context.reserve_tokens)
        
        original_tokens = context.total_tokens
        messages_removed = 0
        messages_summarized = 0
        
        if original_tokens <= target:
            return TruncationResult(
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                messages_removed=0,
                messages_summarized=0,
                strategy_used=strategy,
            )
        
        if strategy == TruncationStrategy.FIFO:
            messages_removed = self._truncate_fifo(context, target)
        
        elif strategy == TruncationStrategy.LIFO:
            messages_removed = self._truncate_lifo(context, target)
        
        elif strategy == TruncationStrategy.IMPORTANCE:
            messages_removed = self._truncate_by_importance(context, target)
        
        elif strategy == TruncationStrategy.SMART:
            messages_removed = self._truncate_smart(context, target)
        
        elif strategy == TruncationStrategy.SUMMARIZE:
            messages_removed, messages_summarized = self._truncate_with_summary(
                context, target
            )
        
        return TruncationResult(
            original_tokens=original_tokens,
            final_tokens=context.total_tokens,
            messages_removed=messages_removed,
            messages_summarized=messages_summarized,
            strategy_used=strategy,
        )
    
    def _truncate_fifo(self, context: ContextWindow, target: int) -> int:
        """Remove oldest messages first."""
        removed = 0
        
        while context.total_tokens > target and context._messages:
            context._messages.pop(0)
            removed += 1
        
        return removed
    
    def _truncate_lifo(self, context: ContextWindow, target: int) -> int:
        """Remove newest messages first."""
        removed = 0
        
        while context.total_tokens > target and context._messages:
            context._messages.pop()
            removed += 1
        
        return removed
    
    def _truncate_by_importance(self, context: ContextWindow, target: int) -> int:
        """Remove least important messages first."""
        removed = 0
        
        while context.total_tokens > target and context._messages:
            # Find least important message
            min_importance = float('inf')
            min_idx = 0
            
            for i, msg in enumerate(context._messages):
                if msg.importance < min_importance:
                    min_importance = msg.importance
                    min_idx = i
            
            context._messages.pop(min_idx)
            removed += 1
        
        return removed
    
    def _truncate_smart(self, context: ContextWindow, target: int) -> int:
        """
        Smart truncation preserving:
        - System message
        - Last N messages
        - High importance messages
        """
        removed = 0
        
        # Keep last 4 messages
        protected_count = 4
        
        while context.total_tokens > target:
            if len(context._messages) <= protected_count:
                break
            
            # Find best message to remove (oldest, lowest importance, not protected)
            candidates = context._messages[:-protected_count]
            
            if not candidates:
                break
            
            # Score each candidate (lower = better to remove)
            best_to_remove = 0
            best_score = float('inf')
            
            for i, msg in enumerate(candidates):
                # Score considers importance and age
                age_factor = i / len(candidates)  # Older = higher
                score = msg.importance * 0.7 + (1 - age_factor) * 0.3
                
                if score < best_score:
                    best_score = score
                    best_to_remove = i
            
            context._messages.pop(best_to_remove)
            removed += 1
        
        return removed
    
    def _truncate_with_summary(
        self,
        context: ContextWindow,
        target: int,
    ) -> Tuple[int, int]:
        """Truncate by summarizing middle messages."""
        removed = 0
        summarized = 0
        
        # Keep first 2 and last 4 messages
        if len(context._messages) <= 6:
            return self._truncate_smart(context, target), 0
        
        # Get messages to summarize
        to_summarize = context._messages[2:-4]
        
        if not to_summarize:
            return 0, 0
        
        # Create simple summary
        summary_content = "Previous conversation summary:\n"
        for msg in to_summarize:
            role = msg.role.value.capitalize()
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_content += f"- {role}: {preview}\n"
        
        summary_message = Message(
            role=MessageRole.SYSTEM,
            content=summary_content,
            importance=0.6,
            content_type=ContentType.TEXT,
        )
        
        # Remove summarized messages and add summary
        new_messages = context._messages[:2] + [summary_message] + context._messages[-4:]
        
        removed = len(to_summarize)
        summarized = len(to_summarize)
        
        context._messages = new_messages
        
        # If still over, apply smart truncation
        if context.total_tokens > target:
            removed += self._truncate_smart(context, target)
        
        return removed, summarized
    
    def fit_content(
        self,
        content: str,
        max_tokens: int,
        preserve_start: int = 100,
        preserve_end: int = 100,
    ) -> str:
        """
        Fit content within token limit, preserving start and end.
        """
        content_tokens = estimate_tokens(content)
        
        if content_tokens <= max_tokens:
            return content
        
        # Calculate how much to keep
        chars_per_token = len(content) / content_tokens
        target_chars = int(max_tokens * chars_per_token)
        
        # Preserve start and end
        start_chars = int(preserve_start * chars_per_token)
        end_chars = int(preserve_end * chars_per_token)
        
        if start_chars + end_chars >= target_chars:
            # Just truncate from end
            return content[:target_chars] + "..."
        
        start_part = content[:start_chars]
        end_part = content[-end_chars:]
        
        return f"{start_part}\n\n... [content truncated] ...\n\n{end_part}"


# Global manager instance
context_manager = ContextWindowManager()


def get_context_manager() -> ContextWindowManager:
    """Get the global context window manager."""
    return context_manager
