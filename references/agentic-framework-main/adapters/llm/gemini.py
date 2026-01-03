"""
Google Gemini LLM adapter.

This adapter integrates with Google's Gemini API for text generation.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole


class GeminiAdapter(LLMAdapter):
    """
    Adapter for Google Gemini API.

    Supports Gemini models including Gemini 2.0, Gemini 1.5 Pro, and others.
    Requires Google Cloud credentials and Gemini API key.
    """

    API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_MODEL: str = "gemini-2.0-flash"
    DEFAULT_MAX_TOKENS: int = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini adapter.

        Args:
            model: Gemini model identifier (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
            api_key: Google Gemini API key (defaults to GEMINI_API_KEY env var)
            **kwargs: Additional configuration

        Raises:
            ValueError: If API key is not provided
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required (set GEMINI_API_KEY env var)")

        super().__init__(model, api_key, **kwargs)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }

        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers=headers,
            timeout=kwargs.get("default_timeout", 60.0),
        )

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert LLMMessage objects to Gemini API format.

        Gemini separates system prompt from messages.

        Args:
            messages: List of LLMMessage objects

        Returns:
            Tuple of (system_prompt, api_messages)
        """
        system_prompt: Optional[str] = None
        api_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                api_messages.append({
                    "role": "user" if msg.role == MessageRole.USER else "model",
                    "content": msg.content,
                })

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
        Generate a completion from Gemini.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters (top_p, top_k, etc.)

        Returns:
            LLM response with content and metadata

        Raises:
            LLMError: If the request fails
            TimeoutError: If the request times out
        """
        try:
            system_prompt, api_messages = self._convert_messages(messages)
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            # Gemini API payload structure
            payload: Dict[str, Any] = {
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
            }

            if system_prompt:
                payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 60.0)

            response = await self.client.post(
                f"chat/completions",
                json=payload,
                timeout=request_timeout,
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", {}).get("message", error_detail)
                except Exception:
                    pass
                raise LLMError(f"Gemini API error ({response.status_code}): {error_detail}")

            response_data = response.json()
            choice = response_data["choices"][0]

            usage = response_data.get("usage", {})

            return LLMResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason", "stop"),
                model=self.model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                raw_response=response_data,
            )

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Gemini request timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Gemini completion failed: {e}")

    async def stream_complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion from Gemini.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Yields:
            Partial response text chunks

        Raises:
            LLMError: If the request fails
        """
        try:
            system_prompt, api_messages = self._convert_messages(messages)
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            payload: Dict[str, Any] = {
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
                "stream": True,
            }

            if system_prompt:
                payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 60.0)

            async with self.client.stream(
                "POST",
                "chat/completions",
                json=payload,
                timeout=request_timeout,
            ) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    raise LLMError(f"Gemini API error ({response.status_code}): {error_detail}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            data = __import__("json").loads(data_str)
                            chunk = data["choices"][0]["delta"].get("content", "")
                            if chunk:
                                yield chunk
                        except Exception:
                            continue

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Gemini request timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Gemini streaming failed: {e}")

    async def validate_api_key(self) -> bool:
        """
        Validate Gemini API key.

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
