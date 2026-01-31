"""
OpenAI LLM adapter.

This adapter integrates with OpenAI's API for GPT models.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI API.

    Supports GPT-4, GPT-3.5, GPT-5, o-series, and other OpenAI models.
    """

    API_BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "gpt-4o"
    DEFAULT_MAX_TOKENS: int = 4096

    # Models that require max_completion_tokens instead of max_tokens
    # Includes: o1, o3, gpt-5.x, and future models
    MODELS_WITH_COMPLETION_TOKENS: tuple = (
        "o1", "o3", "o4",  # Reasoning models
        "gpt-5",  # GPT-5 series
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI adapter.

        Args:
            model: OpenAI model identifier
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")

        super().__init__(model, api_key, **kwargs)
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers=headers,
            timeout=kwargs.get("default_timeout", 60.0),
        )

    def _uses_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        return any(self.model.startswith(prefix) for prefix in self.MODELS_WITH_COMPLETION_TOKENS)

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert LLMMessage objects to OpenAI API format.

        Args:
            messages: List of LLMMessage objects

        Returns:
            List of messages in OpenAI format
        """
        result = []
        for msg in messages:
            # Handle role as either enum or string (due to use_enum_values config)
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            result.append({"role": role, "content": msg.content})
        return result

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion using OpenAI API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLM response

        Raises:
            LLMError: If API request fails
        """
        api_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }

        if max_tokens:
            # Use max_completion_tokens for newer models (o1, o3, gpt-5.x, etc.)
            if self._uses_completion_tokens():
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        # Add any additional parameters
        payload.update(kwargs)

        try:
            response = await self.client.post(
                "/chat/completions", json=payload, timeout=timeout or 60.0
            )
            response.raise_for_status()
            data = response.json()

            # Extract content from response
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")

            return LLMResponse(
                content=content,
                finish_reason=choice.get("finish_reason", "unknown"),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                },
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"OpenAI API request failed: {e.response.text}",
                provider="openai",
                original_error=e,
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error calling OpenAI API: {str(e)}",
                provider="openai",
                original_error=e,
            )

    async def stream_complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate streaming completion using OpenAI API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive

        Raises:
            LLMError: If API request fails
        """
        api_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            # Use max_completion_tokens for newer models (o1, o3, gpt-5.x, etc.)
            if self._uses_completion_tokens():
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload, timeout=timeout or 60.0
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == "[DONE]":
                            break

                        try:
                            import json

                            event = json.loads(data)
                            delta = event.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"OpenAI streaming request failed: {e.response.text}",
                provider="openai",
                original_error=e,
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error in OpenAI streaming: {str(e)}",
                provider="openai",
                original_error=e,
            )

    async def validate_api_key(self) -> bool:
        """
        Validate API key by making a test request.

        Returns:
            True if API key is valid

        Raises:
            LLMError: If validation fails
        """
        try:
            # Make a minimal request to validate the key
            test_messages = [LLMMessage(role=MessageRole.USER, content="test")]
            await self.complete(test_messages, max_tokens=1)
            return True
        except LLMError:
            return False

    async def __aenter__(self) -> "OpenAIAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.client.aclose()
