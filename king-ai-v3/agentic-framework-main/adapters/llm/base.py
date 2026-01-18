"""
Abstract base adapter for LLM providers.

This module defines the contract that all LLM adapters must implement,
ensuring interchangeability across different providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in LLM conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    class Config:
        """Pydantic config."""

        use_enum_values = True


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    finish_reason: str
    model: str
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    tool_calls: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.

    All LLM adapters must implement this interface to ensure interchangeability
    across different providers (Anthropic, OpenAI, Azure, local models, etc.).
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the LLM adapter.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
            api_key: API key for the provider (if required)
            **kwargs: Provider-specific configuration options
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Provider-specific parameters

        Returns:
            LLM response with content and metadata

        Raises:
            LLMError: If the request fails
            TimeoutError: If the request times out
        """
        pass

    @abstractmethod
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate a streaming completion from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Provider-specific parameters

        Yields:
            Partial LLM responses as they arrive

        Raises:
            LLMError: If the request fails
            TimeoutError: If the request times out
        """
        pass

    @abstractmethod
    async def validate_api_key(self) -> bool:
        """
        Validate the API key and connection to the provider.

        Returns:
            True if the API key is valid and the provider is reachable

        Raises:
            LLMError: If validation fails
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Default implementation uses a simple heuristic. Override for
        provider-specific tokenization.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 characters per token
        return len(text) // 4

    def format_system_prompt(self, prompt: str, capabilities: List[str]) -> str:
        """
        Format system prompt with capability context.

        Args:
            prompt: Base system prompt
            capabilities: List of enabled capabilities

        Returns:
            Formatted system prompt
        """
        if not capabilities:
            return prompt

        capability_list = "\n".join(f"- {cap}" for cap in capabilities)
        return f"{prompt}\n\nEnabled capabilities:\n{capability_list}"


class LLMError(Exception):
    """Base exception for LLM adapter errors."""

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        """
        Initialize LLM error.

        Args:
            message: Error message
            provider: Provider name
            original_error: Original exception if wrapping another error
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
