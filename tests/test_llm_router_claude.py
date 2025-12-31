import pytest
from unittest.mock import AsyncMock, patch

from src.utils.llm_router import LLMRouter, TaskContext


@pytest.mark.asyncio
async def test_high_risk_routes_to_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with patch("src.utils.llm_router.OllamaClient") as MockOllama, patch(
        "src.utils.llm_router.ClaudeClient"
    ) as MockClaude:
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
async def test_without_anthropic_key_does_not_use_claude(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("src.utils.llm_router.OllamaClient") as MockOllama:
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
