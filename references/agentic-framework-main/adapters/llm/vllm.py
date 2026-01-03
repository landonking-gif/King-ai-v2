"""
vLLM local inference server adapter.

This adapter integrates with vLLM for high-performance local LLM inference.
vLLM provides OpenAI-compatible API with optimized inference.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole


class VLLMAdapter(LLMAdapter):
    """
    Adapter for vLLM local inference server.

    vLLM provides OpenAI-compatible API with optimized inference.
    Default: http://localhost:8000 (vLLM default port)
    """

    DEFAULT_ENDPOINT: str = "http://localhost:8000"
    DEFAULT_MODEL: str = "meta-llama/Llama-2-70b-hf"
    DEFAULT_MAX_TOKENS: int = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize vLLM adapter.

        Args:
            model: Model identifier (HuggingFace model name or path)
            endpoint: vLLM server endpoint (defaults to VLLM_ENDPOINT env var or http://localhost:8000)
            api_key: Optional API key (if vLLM is behind authentication)
            **kwargs: Additional configuration

        Raises:
            ValueError: If endpoint is unreachable
        """
        endpoint = endpoint or os.getenv("VLLM_ENDPOINT", self.DEFAULT_ENDPOINT)

        super().__init__(model, api_key, **kwargs)

        self.endpoint = endpoint.rstrip("/")

        # Build headers
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers=headers,
            timeout=kwargs.get("default_timeout", 120.0),  # Longer timeout for local inference
        )

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert LLMMessage objects to OpenAI-compatible format.

        Args:
            messages: List of LLMMessage objects

        Returns:
            List of messages in OpenAI format
        """
        result = []
        for msg in messages:
            # Handle role as either enum or string (due to use_enum_values config)
            role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
            msg_dict = {
                "role": role,
                "content": msg.content,
            }
            if msg.name:
                msg_dict["name"] = msg.name
            result.append(msg_dict)
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
        Generate a completion from vLLM server.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters (top_p, top_k, repetition_penalty, etc.)

        Returns:
            LLM response with content and metadata

        Raises:
            LLMError: If the request fails
            TimeoutError: If the request times out
        """
        try:
            api_messages = self._convert_messages(messages)
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            payload = {
                "model": self.model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 1.0),
                "top_k": kwargs.get("top_k", -1),  # -1 means use no top-k filtering
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "stop": kwargs.get("stop"),
                "logprobs": kwargs.get("logprobs"),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 120.0)

            response = await self.client.post(
                "/v1/chat/completions",
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
                raise LLMError(f"vLLM error ({response.status_code}): {error_detail}")

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
            raise TimeoutError(
                f"vLLM request timed out (endpoint: {self.endpoint}). "
                f"Ensure server is running: python -m vllm.entrypoints.openai.api_server"
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"vLLM completion failed: {e}")

    async def stream_complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion from vLLM server.

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
            api_messages = self._convert_messages(messages)
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            payload = {
                "model": self.model,
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "top_p": kwargs.get("top_p", 1.0),
                "top_k": kwargs.get("top_k", -1),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                "stop": kwargs.get("stop"),
            }

            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 120.0)

            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=request_timeout,
            ) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    raise LLMError(f"vLLM error ({response.status_code}): {error_detail}")

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

        except httpx.TimeoutException:
            raise TimeoutError(
                f"vLLM request timed out. Ensure server is running: "
                f"python -m vllm.entrypoints.openai.api_server"
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"vLLM streaming failed: {e}")

    async def validate_api_key(self) -> bool:
        """
        Validate vLLM server connection.

        Returns:
            True if server is reachable

        Raises:
            LLMError: If validation fails
        """
        try:
            # Make a minimal request to validate the connection
            test_messages = [LLMMessage(role=MessageRole.USER, content="test")]
            await self.complete(test_messages, max_tokens=1)
            return True
        except (LLMError, TimeoutError):
            return False
