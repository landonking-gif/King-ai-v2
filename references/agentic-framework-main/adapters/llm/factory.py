"""
LLM Adapter Factory with environment-based defaults.

Module: adapters/llm/factory.py

Provides a unified interface for creating LLM adapters with:
- Environment variable defaults (.env)
- Runtime parameter overrides
- Both sync and async support
- Provider detection from model names
"""

import asyncio
import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from .base import LLMAdapter, LLMError, LLMMessage, LLMResponse, MessageRole

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GEMINI = "gemini"
    LOCAL = "local"
    VLLM = "vllm"
    MOCK = "mock"


# Model prefix to provider mapping
MODEL_PROVIDER_MAP = {
    "gpt-": LLMProvider.OPENAI,
    "o1": LLMProvider.OPENAI,
    "o3": LLMProvider.OPENAI,
    "claude-": LLMProvider.ANTHROPIC,
    "gemini-": LLMProvider.GEMINI,
    "llama": LLMProvider.LOCAL,
    "mistral": LLMProvider.LOCAL,
    "codellama": LLMProvider.LOCAL,
    "phi": LLMProvider.LOCAL,
    "qwen": LLMProvider.LOCAL,
}

# Environment variable mapping for each provider
PROVIDER_ENV_MAP = {
    LLMProvider.OPENAI: {
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
        "default_model": "gpt-4o",
    },
    LLMProvider.ANTHROPIC: {
        "api_key": "ANTHROPIC_API_KEY",
        "model": "ANTHROPIC_MODEL",
        "default_model": "claude-sonnet-4-20250514",
    },
    LLMProvider.AZURE: {
        "api_key": "AZURE_OPENAI_KEY",
        "model": "AZURE_MODEL",
        "endpoint": "AZURE_OPENAI_ENDPOINT",
        "default_model": "gpt-4o",
    },
    LLMProvider.GEMINI: {
        "api_key": "GEMINI_API_KEY",
        "model": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
    },
    LLMProvider.LOCAL: {
        "api_key": None,  # No API key needed
        "model": "LOCAL_MODEL",
        "endpoint": "OLLAMA_ENDPOINT",
        "default_model": "llama3.1:70b",
        "default_endpoint": "http://localhost:11434",
    },
    LLMProvider.VLLM: {
        "api_key": None,
        "model": "VLLM_MODEL",
        "endpoint": "VLLM_ENDPOINT",
        "default_model": "meta-llama/Llama-2-70b-hf",
        "default_endpoint": "http://localhost:8000",
    },
}


def _find_and_load_env() -> None:
    """Find and load .env file from project root."""
    current = Path(__file__).resolve().parent

    # Search upward for .env
    for _ in range(10):
        env_path = current / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return

        # Check for project markers
        markers = [".git", "pyproject.toml", "setup.py", "CLAUDE.md"]
        for marker in markers:
            if (current / marker).exists():
                if env_path.exists():
                    load_dotenv(env_path, override=True)
                    return
                parent_env = current.parent / ".env"
                if parent_env.exists():
                    load_dotenv(parent_env, override=True)
                    return

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback to default load_dotenv
    load_dotenv(override=True)


# Load .env on module import
_find_and_load_env()


def detect_provider(model: str) -> LLMProvider:
    """
    Detect provider from model name.

    Args:
        model: Model identifier

    Returns:
        Detected provider enum
    """
    model_lower = model.lower()

    for prefix, provider in MODEL_PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return provider

    # Check for specific patterns
    if "text-embedding" in model_lower:
        return LLMProvider.OPENAI

    # Default to OpenAI
    logger.warning(f"Could not detect provider for model '{model}', defaulting to OpenAI")
    return LLMProvider.OPENAI


def get_default_provider() -> LLMProvider:
    """
    Get the default provider based on available API keys.

    Priority:
    1. LLM_PROVIDER env var if set
    2. First provider with available API key
    3. OpenAI as fallback

    Returns:
        Default provider
    """
    # Check explicit provider setting
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower()
    if explicit_provider:
        try:
            return LLMProvider(explicit_provider)
        except ValueError:
            logger.warning(f"Invalid LLM_PROVIDER '{explicit_provider}', detecting from API keys")

    # Detect from available API keys
    if os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC
    if os.getenv("AZURE_OPENAI_KEY"):
        return LLMProvider.AZURE
    if os.getenv("GEMINI_API_KEY"):
        return LLMProvider.GEMINI

    # Check for local providers
    if os.getenv("OLLAMA_ENDPOINT"):
        return LLMProvider.LOCAL
    if os.getenv("VLLM_ENDPOINT"):
        return LLMProvider.VLLM

    # Default
    return LLMProvider.OPENAI


def get_default_model(provider: Optional[LLMProvider] = None) -> str:
    """
    Get the default model for a provider.

    Args:
        provider: Provider to get model for (defaults to default provider)

    Returns:
        Default model name
    """
    if provider is None:
        provider = get_default_provider()

    env_config = PROVIDER_ENV_MAP.get(provider, {})
    model_env_var = env_config.get("model")

    if model_env_var:
        env_model = os.getenv(model_env_var)
        if env_model:
            # Clean up potential whitespace from .env
            return env_model.strip().replace(" ", "")

    return env_config.get("default_model", "gpt-4o")


def get_embedding_model() -> str:
    """
    Get the embedding model from environment.

    Returns:
        Embedding model name
    """
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def create_adapter(
    provider: Optional[Union[str, LLMProvider]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMAdapter:
    """
    Create an LLM adapter with environment defaults and runtime overrides.

    Priority (highest to lowest):
    1. Explicit parameters passed to this function
    2. Environment variables from .env
    3. Built-in defaults

    Args:
        provider: LLM provider (openai, anthropic, azure, gemini, local, vllm)
        model: Model identifier (overrides env default)
        api_key: API key (overrides env default)
        **kwargs: Additional provider-specific options

    Returns:
        Configured LLM adapter instance

    Example:
        # Use .env defaults
        adapter = create_adapter()

        # Override model
        adapter = create_adapter(model="gpt-4o-mini")

        # Force specific provider
        adapter = create_adapter(provider="anthropic", model="claude-haiku-4-20250514")
    """
    # Determine provider
    if provider is None:
        if model:
            provider = detect_provider(model)
        else:
            provider = get_default_provider()
    elif isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    # Get model
    if model is None:
        model = get_default_model(provider)

    # Get provider config
    env_config = PROVIDER_ENV_MAP.get(provider, {})

    # Get API key
    if api_key is None and env_config.get("api_key"):
        api_key = os.getenv(env_config["api_key"])

    logger.info(f"Creating {provider.value} adapter with model: {model}")

    # Create adapter
    if provider == LLMProvider.OPENAI:
        from .openai import OpenAIAdapter

        return OpenAIAdapter(model=model, api_key=api_key, **kwargs)

    elif provider == LLMProvider.ANTHROPIC:
        from .anthropic import AnthropicAdapter

        return AnthropicAdapter(model=model, api_key=api_key, **kwargs)

    elif provider == LLMProvider.AZURE:
        from .azure import AzureOpenAIAdapter

        endpoint = kwargs.pop("endpoint", None) or os.getenv(
            env_config.get("endpoint", ""), ""
        )
        return AzureOpenAIAdapter(
            model=model, api_key=api_key, endpoint=endpoint, **kwargs
        )

    elif provider == LLMProvider.GEMINI:
        from .gemini import GeminiAdapter

        return GeminiAdapter(model=model, api_key=api_key, **kwargs)

    elif provider == LLMProvider.LOCAL:
        from .local import LocalLLMAdapter

        endpoint = kwargs.pop("endpoint", None) or os.getenv(
            env_config.get("endpoint", ""),
            env_config.get("default_endpoint", "http://localhost:11434"),
        )
        return LocalLLMAdapter(model=model, endpoint=endpoint, **kwargs)

    elif provider == LLMProvider.VLLM:
        from .vllm import VLLMAdapter

        endpoint = kwargs.pop("endpoint", None) or os.getenv(
            env_config.get("endpoint", ""),
            env_config.get("default_endpoint", "http://localhost:8000"),
        )
        return VLLMAdapter(model=model, endpoint=endpoint, **kwargs)

    elif provider == LLMProvider.MOCK:
        from .mock import MockLLMAdapter

        return MockLLMAdapter(model=model, **kwargs)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


class SyncLLMWrapper:
    """
    Synchronous wrapper for async LLM adapters.

    Used by Haystack components and other sync code that needs LLM access.
    """

    def __init__(self, adapter: LLMAdapter):
        """
        Initialize sync wrapper.

        Args:
            adapter: Async LLM adapter to wrap
        """
        self.adapter = adapter
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing event loop or create a new one."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to use run_coroutine_threadsafe
                return loop
            return loop
        except RuntimeError:
            # No event loop in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - use nest_asyncio or thread
                import concurrent.futures
                import threading

                result = None
                exception = None

                def run_in_thread() -> None:
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(coro)
                        finally:
                            new_loop.close()
                    except Exception as e:
                        exception = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=120)  # 2 minute timeout

                if exception:
                    raise exception
                return result
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion synchronously.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        return self._run_async(
            self.adapter.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        )

    def complete_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple text completion.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        messages = [LLMMessage(role=MessageRole.USER, content=prompt)]
        response = self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Chat completion with dict messages (for compatibility).

        Args:
            messages: List of {"role": str, "content": str} dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Dict with "content" and "usage" keys
        """
        llm_messages = [
            LLMMessage(role=MessageRole(m["role"]), content=m["content"])
            for m in messages
        ]
        response = self.complete(
            messages=llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return {
            "content": response.content,
            "usage": response.usage,
            "model": response.model,
        }

    @property
    def model(self) -> str:
        """Get model name."""
        return self.adapter.model


def create_sync_adapter(
    provider: Optional[Union[str, LLMProvider]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> SyncLLMWrapper:
    """
    Create a synchronous LLM adapter wrapper.

    Same parameters as create_adapter().

    Returns:
        SyncLLMWrapper instance for synchronous usage
    """
    adapter = create_adapter(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs,
    )
    return SyncLLMWrapper(adapter)


# Convenience functions for getting configuration
def get_llm_config() -> Dict[str, Any]:
    """
    Get current LLM configuration from environment.

    Returns:
        Dict with provider, model, and embedding_model
    """
    provider = get_default_provider()
    return {
        "provider": provider.value,
        "model": get_default_model(provider),
        "embedding_model": get_embedding_model(),
        "api_key_set": bool(
            os.getenv(PROVIDER_ENV_MAP.get(provider, {}).get("api_key", ""))
        ),
    }


# Export key classes and functions
__all__ = [
    "LLMProvider",
    "create_adapter",
    "create_sync_adapter",
    "detect_provider",
    "get_default_provider",
    "get_default_model",
    "get_embedding_model",
    "get_llm_config",
    "SyncLLMWrapper",
]
