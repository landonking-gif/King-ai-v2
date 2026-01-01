import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.utils.llm_router import LLMRouter, TaskContext


@pytest.mark.asyncio
async def test_high_risk_routes_to_claude():
    with patch("src.utils.llm_router.settings") as mock_settings, \
         patch("src.utils.llm_router.OllamaClient") as MockOllama, \
         patch("src.utils.llm_router.ClaudeClient") as MockClaude:
        
        # Configure mock settings
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.1:8b"
        mock_settings.vllm_url = None
        mock_settings.vllm_model = "meta-llama/Llama-3.1-70B-Instruct"
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.claude_model = "claude-3-5-sonnet-20241022"
        mock_settings.gemini_api_key = None
        mock_settings.gemini_api_keys = ""
        
        MockOllama.return_value.complete = AsyncMock(return_value="ollama-response")

        claude_instance = MockClaude.return_value
        claude_instance.is_available.return_value = True
        claude_instance.complete = AsyncMock(return_value="claude-response")

        router = LLMRouter()
        context = TaskContext(
            task_type="legal",
            risk_level="high",
            requires_accuracy=True,
            token_estimate=128,
            priority="critical",
        )

        result = await router.complete("hello", system=None, context=context)
        assert result == "claude-response"
        claude_instance.complete.assert_awaited()


@pytest.mark.asyncio
async def test_without_anthropic_key_does_not_use_claude():
    with patch("src.utils.llm_router.settings") as mock_settings, \
         patch("src.utils.llm_router.OllamaClient") as MockOllama:
        
        # Configure mock settings without Anthropic key
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.1:8b"
        mock_settings.vllm_url = None
        mock_settings.vllm_model = "meta-llama/Llama-3.1-70B-Instruct"
        mock_settings.anthropic_api_key = None
        mock_settings.claude_model = "claude-3-5-sonnet-20241022"
        mock_settings.gemini_api_key = None
        mock_settings.gemini_api_keys = ""
        
        MockOllama.return_value.complete = AsyncMock(return_value="ollama-response")

        router = LLMRouter()
        context = TaskContext(
            task_type="content",
            risk_level="low",
            requires_accuracy=False,
            token_estimate=128,
            priority="normal",
        )

        result = await router.complete("hello", system=None, context=context)
        assert result == "ollama-response"
