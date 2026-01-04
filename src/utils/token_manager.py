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
    SOURCE_CODE = "source_code"               # Priority 3 (for recursive development)
    RELEVANT_MEMORY = "relevant_memory"       # Priority 4
    RECENT_CONVERSATION = "recent_conversation"  # Priority 5
    BUSINESS_DATA = "business_data"           # Priority 6
    TASK_HISTORY = "task_history"             # Priority 7
    FULL_HISTORY = "full_history"             # Priority 8 (lowest)


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
            ContextSection.SOURCE_CODE: (0.20, 3),         # 20% - for recursive development
            ContextSection.RELEVANT_MEMORY: (0.15, 4),     # 15% - RAG context
            ContextSection.RECENT_CONVERSATION: (0.15, 5), # 15% - recent turns
            ContextSection.BUSINESS_DATA: (0.15, 6),       # 15% - business info
            ContextSection.TASK_HISTORY: (0.10, 7),        # 10% - task context
            ContextSection.FULL_HISTORY: (0.05, 8),        # 5% - historical
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
