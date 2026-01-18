"""
Mock LLM adapter for testing and development.

This adapter simulates LLM responses without making actual API calls,
useful for testing and development.
"""

from typing import Any, AsyncIterator, List, Optional

import anyio

from adapters.llm.base import LLMAdapter, LLMMessage, LLMResponse


class MockLLMAdapter(LLMAdapter):
    """
    Mock LLM adapter for testing.

    Returns predefined or generated responses without calling external APIs.
    """

    def __init__(
        self,
        model: str = "mock-model",
        api_key: Optional[str] = None,
        response_template: str = "Mock response to: {prompt}",
        delay_ms: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize mock adapter.

        Args:
            model: Mock model identifier
            api_key: Not used, but accepted for interface compatibility
            response_template: Template for generating responses
            delay_ms: Simulated latency in milliseconds
            **kwargs: Additional configuration
        """
        super().__init__(model, api_key, **kwargs)
        self.response_template = response_template
        self.delay_ms = delay_ms
        self.call_count = 0

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a mock completion.

        Args:
            messages: Conversation history
            temperature: Not used in mock
            max_tokens: Not used in mock
            timeout: Not used in mock
            **kwargs: Additional parameters

        Returns:
            Mock LLM response
        """
        # Simulate network delay
        await anyio.sleep(self.delay_ms / 1000.0)

        self.call_count += 1

        # Get last user message
        user_messages = [msg for msg in messages if msg.role == "user"]
        last_prompt = user_messages[-1].content if user_messages else "no prompt"

        # Generate mock response
        content = self.response_template.format(prompt=last_prompt[:50])

        # Calculate mock token usage
        prompt_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        completion_tokens = self.count_tokens(content)

        return LLMResponse(
            content=content,
            finish_reason="stop",
            model=self.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            raw_response={
                "mock": True,
                "call_count": self.call_count,
                "temperature": temperature,
            },
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
        Generate a streaming mock completion.

        Args:
            messages: Conversation history
            temperature: Not used in mock
            max_tokens: Not used in mock
            timeout: Not used in mock
            **kwargs: Additional parameters

        Yields:
            Chunks of the mock response
        """
        # Get full response
        response = await self.complete(messages, temperature, max_tokens, timeout, **kwargs)

        # Stream it word by word
        words = response.content.split()
        for word in words:
            await anyio.sleep(self.delay_ms / 1000.0 / len(words))
            yield word + " "

    async def validate_api_key(self) -> bool:
        """
        Validate mock API key (always succeeds).

        Returns:
            Always True
        """
        await anyio.sleep(0.01)  # Simulate validation delay
        return True
