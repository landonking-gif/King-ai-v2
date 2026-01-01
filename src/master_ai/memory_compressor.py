"""
Conversation Memory Compressor.
Summarizes and compresses conversation history to reduce token usage.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("memory_compressor")


class MessageRole(str, Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A conversation message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    importance: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class ConversationSummary:
    """A summary of compressed messages."""
    content: str
    original_count: int
    original_tokens: int
    compressed_tokens: int
    start_time: datetime
    end_time: datetime
    key_points: List[str] = field(default_factory=list)


class CompressionStrategy(str, Enum):
    """Strategy for compressing conversations."""
    TRUNCATE = "truncate"  # Remove oldest messages
    SUMMARIZE = "summarize"  # Summarize old messages
    SLIDING_WINDOW = "sliding_window"  # Keep recent + important
    HIERARCHICAL = "hierarchical"  # Create nested summaries


@dataclass
class CompressionConfig:
    """Configuration for memory compression."""
    max_tokens: int = 8000
    target_tokens: int = 6000
    min_messages_to_keep: int = 5
    summarize_threshold: int = 10  # Messages before summarizing
    strategy: CompressionStrategy = CompressionStrategy.SLIDING_WINDOW
    importance_threshold: float = 0.3  # Keep messages above this
    summary_max_tokens: int = 500


class TokenEstimator:
    """Estimates token counts for text."""
    
    # Approximate tokens per character for English text
    TOKENS_PER_CHAR = 0.25
    
    @classmethod
    def estimate(cls, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        # More accurate estimation considering:
        # - Whitespace (roughly 1 token per word)
        # - Punctuation (often separate tokens)
        # - Code blocks (more tokens)
        
        word_count = len(text.split())
        char_count = len(text)
        
        # Base estimate from words
        word_estimate = word_count * 1.3
        
        # Adjust for character count
        char_estimate = char_count * cls.TOKENS_PER_CHAR
        
        # Use the higher estimate
        return max(int(word_estimate), int(char_estimate))
    
    @classmethod
    def estimate_messages(cls, messages: List[Message]) -> int:
        """Estimate total tokens for a list of messages."""
        total = 0
        for msg in messages:
            if msg.token_count > 0:
                total += msg.token_count
            else:
                total += cls.estimate(msg.content)
                # Add overhead for role and formatting
                total += 4
        return total


class MemoryCompressor:
    """
    Compresses conversation memory to fit within token limits.
    
    Features:
    - Multiple compression strategies
    - Importance-based retention
    - Automatic summarization
    - Token counting and management
    """
    
    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        summarizer = None,  # LLM client for summarization
    ):
        self.config = config or CompressionConfig()
        self._summarizer = summarizer
        self._summaries: List[ConversationSummary] = []
    
    def set_summarizer(self, summarizer) -> None:
        """Set the LLM client for summarization."""
        self._summarizer = summarizer
    
    async def compress(
        self,
        messages: List[Message],
        target_tokens: Optional[int] = None,
    ) -> Tuple[List[Message], int]:
        """
        Compress conversation to fit within token limit.
        
        Args:
            messages: List of conversation messages
            target_tokens: Target token count (default from config)
            
        Returns:
            Tuple of (compressed messages, token count)
        """
        target = target_tokens or self.config.target_tokens
        
        # Calculate current token count
        current_tokens = TokenEstimator.estimate_messages(messages)
        
        if current_tokens <= target:
            # No compression needed
            return messages, current_tokens
        
        # Apply compression strategy
        if self.config.strategy == CompressionStrategy.TRUNCATE:
            return await self._truncate(messages, target)
        elif self.config.strategy == CompressionStrategy.SUMMARIZE:
            return await self._summarize(messages, target)
        elif self.config.strategy == CompressionStrategy.SLIDING_WINDOW:
            return await self._sliding_window(messages, target)
        elif self.config.strategy == CompressionStrategy.HIERARCHICAL:
            return await self._hierarchical(messages, target)
        else:
            return await self._truncate(messages, target)
    
    async def _truncate(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> Tuple[List[Message], int]:
        """Simple truncation - remove oldest messages."""
        if not messages:
            return [], 0
        
        # Always keep system message if present
        system_msg = None
        other_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and system_msg is None:
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # Keep most recent messages
        result = []
        current_tokens = 0
        
        if system_msg:
            current_tokens = TokenEstimator.estimate(system_msg.content) + 4
            result.append(system_msg)
        
        # Add messages from most recent
        for msg in reversed(other_messages):
            msg_tokens = TokenEstimator.estimate(msg.content) + 4
            if current_tokens + msg_tokens <= target_tokens:
                result.insert(1 if system_msg else 0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result, current_tokens
    
    async def _summarize(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> Tuple[List[Message], int]:
        """Summarize older messages into a summary message."""
        if not self._summarizer:
            # Fall back to truncation
            return await self._truncate(messages, target_tokens)
        
        # Separate system message
        system_msg = None
        other_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and system_msg is None:
                system_msg = msg
            else:
                other_messages.append(msg)
        
        if len(other_messages) <= self.config.min_messages_to_keep:
            return messages, TokenEstimator.estimate_messages(messages)
        
        # Determine how many messages to summarize
        messages_to_summarize = other_messages[:-self.config.min_messages_to_keep]
        messages_to_keep = other_messages[-self.config.min_messages_to_keep:]
        
        # Generate summary
        summary = await self._generate_summary(messages_to_summarize)
        
        # Create summary message
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Previous conversation summary: {summary.content}]",
            importance=0.8,
            metadata={"is_summary": True},
        )
        
        # Build result
        result = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(messages_to_keep)
        
        self._summaries.append(summary)
        
        return result, TokenEstimator.estimate_messages(result)
    
    async def _sliding_window(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> Tuple[List[Message], int]:
        """Keep recent messages plus high-importance ones."""
        if not messages:
            return [], 0
        
        # Separate by role and importance
        system_msg = None
        high_importance = []
        other_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and system_msg is None:
                system_msg = msg
            elif msg.importance >= self.config.importance_threshold:
                high_importance.append(msg)
            else:
                other_messages.append(msg)
        
        result = []
        current_tokens = 0
        
        # Always include system message
        if system_msg:
            current_tokens = TokenEstimator.estimate(system_msg.content) + 4
            result.append(system_msg)
        
        # Add high importance messages
        for msg in high_importance:
            msg_tokens = TokenEstimator.estimate(msg.content) + 4
            if current_tokens + msg_tokens <= target_tokens * 0.3:  # Reserve 30% for important
                result.append(msg)
                current_tokens += msg_tokens
        
        # Fill remaining with recent messages
        remaining_budget = target_tokens - current_tokens
        
        for msg in reversed(other_messages):
            msg_tokens = TokenEstimator.estimate(msg.content) + 4
            if msg_tokens <= remaining_budget:
                result.append(msg)
                current_tokens += msg_tokens
                remaining_budget -= msg_tokens
        
        # Sort by timestamp to maintain order
        if system_msg:
            non_system = [m for m in result if m.role != MessageRole.SYSTEM]
            non_system.sort(key=lambda m: m.timestamp)
            result = [system_msg] + non_system
        else:
            result.sort(key=lambda m: m.timestamp)
        
        return result, current_tokens
    
    async def _hierarchical(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> Tuple[List[Message], int]:
        """Create nested summaries at different time scales."""
        if not self._summarizer:
            return await self._sliding_window(messages, target_tokens)
        
        # Group messages by time periods
        now = datetime.utcnow()
        recent = []  # Last hour
        older = []   # Last day
        oldest = []  # Before that
        
        for msg in messages:
            age_hours = (now - msg.timestamp).total_seconds() / 3600
            if age_hours < 1:
                recent.append(msg)
            elif age_hours < 24:
                older.append(msg)
            else:
                oldest.append(msg)
        
        result = []
        current_tokens = 0
        
        # Summarize oldest if needed
        if oldest:
            summary = await self._generate_summary(oldest)
            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[Historical context: {summary.content}]",
                importance=0.7,
            )
            result.append(summary_msg)
            current_tokens += TokenEstimator.estimate(summary_msg.content) + 4
        
        # Add older messages or summarize
        if older:
            older_tokens = TokenEstimator.estimate_messages(older)
            if older_tokens > target_tokens * 0.3:
                summary = await self._generate_summary(older)
                summary_msg = Message(
                    role=MessageRole.SYSTEM,
                    content=f"[Recent context: {summary.content}]",
                    importance=0.8,
                )
                result.append(summary_msg)
                current_tokens += TokenEstimator.estimate(summary_msg.content) + 4
            else:
                result.extend(older)
                current_tokens += older_tokens
        
        # Add recent messages
        for msg in recent:
            msg_tokens = TokenEstimator.estimate(msg.content) + 4
            if current_tokens + msg_tokens <= target_tokens:
                result.append(msg)
                current_tokens += msg_tokens
        
        return result, current_tokens
    
    async def _generate_summary(
        self,
        messages: List[Message],
    ) -> ConversationSummary:
        """Generate a summary of messages using LLM."""
        # Build conversation text
        conv_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in messages
        ])
        
        # Generate summary prompt
        prompt = f"""Summarize the following conversation concisely, capturing:
1. Main topics discussed
2. Key decisions or conclusions
3. Important context for future reference

Conversation:
{conv_text}

Summary:"""
        
        try:
            # Call summarizer
            if hasattr(self._summarizer, 'generate'):
                summary_text = await self._summarizer.generate(
                    prompt,
                    max_tokens=self.config.summary_max_tokens,
                )
            else:
                # Fallback to simple extraction
                summary_text = self._simple_summary(messages)
            
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary_text = self._simple_summary(messages)
        
        return ConversationSummary(
            content=summary_text,
            original_count=len(messages),
            original_tokens=TokenEstimator.estimate_messages(messages),
            compressed_tokens=TokenEstimator.estimate(summary_text),
            start_time=messages[0].timestamp if messages else datetime.utcnow(),
            end_time=messages[-1].timestamp if messages else datetime.utcnow(),
        )
    
    def _simple_summary(self, messages: List[Message]) -> str:
        """Create a simple extractive summary."""
        # Extract key sentences (first sentence from each message)
        key_points = []
        
        for msg in messages[:10]:  # Limit to first 10
            sentences = msg.content.split('.')
            if sentences:
                first_sentence = sentences[0].strip()
                if first_sentence and len(first_sentence) > 10:
                    key_points.append(f"- {first_sentence}")
        
        return "Key points:\n" + "\n".join(key_points[:5])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        total_original = sum(s.original_tokens for s in self._summaries)
        total_compressed = sum(s.compressed_tokens for s in self._summaries)
        
        return {
            "summaries_created": len(self._summaries),
            "total_original_tokens": total_original,
            "total_compressed_tokens": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 1.0,
            "tokens_saved": total_original - total_compressed,
        }


def mark_important(
    message: Message,
    importance: float = 1.0,
    reason: Optional[str] = None,
) -> Message:
    """
    Mark a message as important for retention.
    
    Usage:
        msg = mark_important(msg, importance=0.9, reason="Contains key decision")
    """
    message.importance = importance
    if reason:
        message.metadata["importance_reason"] = reason
    return message


# Global compressor instance
memory_compressor = MemoryCompressor()


def get_memory_compressor() -> MemoryCompressor:
    """Get the global memory compressor instance."""
    return memory_compressor
