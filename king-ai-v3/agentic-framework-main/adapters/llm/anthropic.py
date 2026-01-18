"""
Anthropic Claude LLM adapter.

This adapter integrates with Anthropic's Claude API for production use.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from pydantic import Field

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole


class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude API.

    Supports all Claude models including Opus, Sonnet, and Haiku variants.
    """

    API_BASE_URL: str = "https://api.anthropic.com/v1"
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS: int = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_version: str = "2023-06-01",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic adapter.

        Args:
            model: Claude model identifier
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            api_version: API version header
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")

        super().__init__(model, api_key, **kwargs)
        self.api_version = api_version
        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": self.api_version,
                "content-type": "application/json",
            },
            timeout=kwargs.get("default_timeout", 60.0),
        )

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert LLMMessage objects to Anthropic API format.

        Args:
            messages: List of LLMMessage objects

        Returns:
            Tuple of (system_prompt, api_messages)
        """
        system_prompt: Optional[str] = None
        api_messages: List[Dict[str, Any]] = []

        for msg in messages:
            # Handle role as either enum or string (due to use_enum_values config)
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role

            if role == MessageRole.SYSTEM.value or msg.role == MessageRole.SYSTEM:
                # Anthropic uses separate system parameter
                system_prompt = msg.content
            else:
                api_messages.append({"role": role, "content": msg.content})

        return system_prompt, api_messages

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion using Anthropic API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLM response

        Raises:
            LLMError: If API request fails
        """
        system_prompt, api_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Add any additional parameters
        payload.update(kwargs)

        try:
            response = await self.client.post(
                "/messages", json=payload, timeout=timeout or 60.0
            )
            response.raise_for_status()
            data = response.json()

            # Extract content from response
            content_blocks = data.get("content", [])
            content = ""
            if content_blocks and isinstance(content_blocks, list):
                content = content_blocks[0].get("text", "")

            return LLMResponse(
                content=content,
                finish_reason=data.get("stop_reason", "unknown"),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": (
                        data.get("usage", {}).get("input_tokens", 0)
                        + data.get("usage", {}).get("output_tokens", 0)
                    ),
                },
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Anthropic API request failed: {e.response.text}",
                provider="anthropic",
                original_error=e,
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error calling Anthropic API: {str(e)}",
                provider="anthropic",
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
        Generate streaming completion using Anthropic API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive

        Raises:
            LLMError: If API request fails
        """
        system_prompt, api_messages = self._convert_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        payload.update(kwargs)

        try:
            async with self.client.stream(
                "POST", "/messages", json=payload, timeout=timeout or 60.0
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
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Anthropic streaming request failed: {e.response.text}",
                provider="anthropic",
                original_error=e,
            )
        except Exception as e:
            raise LLMError(
                f"Unexpected error in Anthropic streaming: {str(e)}",
                provider="anthropic",
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

    async def __aenter__(self) -> "AnthropicAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.client.aclose()
