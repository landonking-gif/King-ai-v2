"""
Azure OpenAI LLM adapter.

This adapter integrates with Azure OpenAI services for GPT models.
Requires Azure OpenAI resource deployment.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole


class AzureOpenAIAdapter(LLMAdapter):
    """
    Adapter for Azure OpenAI API.

    Supports GPT-4, GPT-3.5, and other Azure-deployed OpenAI models.
    Requires Azure OpenAI resource setup with model deployments.
    """

    DEFAULT_MODEL: str = "gpt-4o"
    DEFAULT_MAX_TOKENS: int = 4096
    DEFAULT_API_VERSION: str = "2024-08-01-preview"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        deployment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Azure OpenAI adapter.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY env var)
            endpoint: Azure resource endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: Azure API version
            deployment_name: Azure deployment name (defaults to model name)
            **kwargs: Additional configuration

        Raises:
            ValueError: If required credentials are missing
        """
        api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        if not api_key:
            raise ValueError("Azure OpenAI API key required (set AZURE_OPENAI_KEY env var)")

        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint required (set AZURE_OPENAI_ENDPOINT env var)")

        super().__init__(model, api_key, **kwargs)

        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.deployment_name = deployment_name or model

        # Azure-specific headers
        headers = {
            "api-key": self.api_key or "",
            "Content-Type": "application/json",
        }

        self.client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers=headers,
            timeout=kwargs.get("default_timeout", 60.0),
        )

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert LLMMessage objects to Azure OpenAI API format.

        Args:
            messages: List of LLMMessage objects

        Returns:
            List of messages in Azure OpenAI format
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
        Generate a completion from Azure OpenAI.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters (top_p, stop, etc.)

        Returns:
            LLM response with content and metadata

        Raises:
            LLMError: If the request fails
        """
        try:
            api_messages = self._convert_messages(messages)
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

            # Azure OpenAI API endpoint format
            url = (
                f"/openai/deployments/{self.deployment_name}/chat/completions"
                f"?api-version={self.api_version}"
            )

            payload = {
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 1.0),
                "stop": kwargs.get("stop"),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 60.0)

            response = await self.client.post(
                url,
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
                raise LLMError(
                    f"Azure OpenAI API error ({response.status_code}): {error_detail}"
                )

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
            raise TimeoutError(f"Azure OpenAI request timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Azure OpenAI completion failed: {e}")

    async def stream_complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion from Azure OpenAI.

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

            url = (
                f"/openai/deployments/{self.deployment_name}/chat/completions"
                f"?api-version={self.api_version}"
            )

            payload = {
                "messages": api_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "top_p": kwargs.get("top_p", 1.0),
                "stop": kwargs.get("stop"),
            }

            payload = {k: v for k, v in payload.items() if v is not None}

            request_timeout = timeout or self.config.get("default_timeout", 60.0)

            async with self.client.stream(
                "POST",
                url,
                json=payload,
                timeout=request_timeout,
            ) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    raise LLMError(
                        f"Azure OpenAI API error ({response.status_code}): {error_detail}"
                    )

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
            raise TimeoutError(f"Azure OpenAI request timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Azure OpenAI streaming failed: {e}")

    async def validate_api_key(self) -> bool:
        """
        Validate Azure OpenAI API key and endpoint.

        Returns:
            True if API key and endpoint are valid

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
