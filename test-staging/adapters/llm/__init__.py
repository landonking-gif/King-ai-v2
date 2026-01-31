"""
LLM adapter implementations for various providers.

Supported adapters:
- AnthropicAdapter: Anthropic Claude API
- OpenAIAdapter: OpenAI GPT models
- AzureOpenAIAdapter: Azure OpenAI services
- GeminiAdapter: Google Gemini API
- LocalLLMAdapter: Ollama and compatible local servers
- VLLMAdapter: vLLM local inference server
- MockLLMAdapter: Mock adapter for testing

Factory functions:
- create_adapter(): Create async adapter with .env defaults
- create_sync_adapter(): Create sync wrapper with .env defaults
- get_default_provider(): Get provider from .env
- get_default_model(): Get model from .env
- get_embedding_model(): Get embedding model from .env
"""

from adapters.llm.base import LLMAdapter, LLMError, LLMResponse, LLMMessage, MessageRole
from adapters.llm.anthropic import AnthropicAdapter
from adapters.llm.openai import OpenAIAdapter
from adapters.llm.azure import AzureOpenAIAdapter
from adapters.llm.gemini import GeminiAdapter
from adapters.llm.local import LocalLLMAdapter
from adapters.llm.vllm import VLLMAdapter
from adapters.llm.mock import MockLLMAdapter
from adapters.llm.factory import (
    LLMProvider,
    create_adapter,
    create_sync_adapter,
    detect_provider,
    get_default_provider,
    get_default_model,
    get_embedding_model,
    get_llm_config,
    SyncLLMWrapper,
)

__all__ = [
    # Base classes
    "LLMAdapter",
    "LLMError",
    "LLMResponse",
    "LLMMessage",
    "MessageRole",
    # Adapters
    "AnthropicAdapter",
    "OpenAIAdapter",
    "AzureOpenAIAdapter",
    "GeminiAdapter",
    "LocalLLMAdapter",
    "VLLMAdapter",
    "MockLLMAdapter",
    # Factory
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
