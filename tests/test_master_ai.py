"""
Unit tests for the Master AI module.
Tests intent classification, planning logic, and safe-guards.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.master_ai.brain import MasterAI
from src.master_ai.planner import Planner
from src.master_ai.models import IntentType, ActionType

# --- Fixtures ---
@pytest.fixture
def mock_llm_router():
    with patch('src.master_ai.brain.LLMRouter') as mock:
        router_instance = mock.return_value
        # Default behavior: return empty JSON object to avoid parsing errors
        router_instance.complete = AsyncMock(return_value='{}')
        yield router_instance

@pytest.fixture
def master_ai(mock_llm_router):
    with patch('src.master_ai.brain.ContextManager'):
        with patch('src.master_ai.brain.Planner'):
            with patch('src.master_ai.brain.EvolutionEngine'):
                with patch('src.master_ai.brain.AgentRouter'):
                    return MasterAI()

# --- Intent Classification Tests ---

@pytest.mark.asyncio
async def test_classify_intent_conversation(master_ai, mock_llm_router):
    """Test that greetings are classified as conversation."""
    # Mock LLM response for classification
    mock_llm_router.complete = AsyncMock(return_value='{"type": "conversation", "action": null, "parameters": {}, "confidence": 0.9}')
    
    result = await master_ai._classify_intent("Hello King AI!", "context")
    assert result.type == IntentType.CONVERSATION

@pytest.mark.asyncio
async def test_classify_intent_command(master_ai, mock_llm_router):
    """Test that explicit instructions are classified as commands."""
    mock_llm_router.complete = AsyncMock(return_value='{"type": "command", "action": "start_business", "parameters": {"niche": "pet toys"}, "confidence": 0.85}')
    
    result = await master_ai._classify_intent("Start a pet toy dropshipping business", "context")
    assert result.type == IntentType.COMMAND
    assert result.action == ActionType.START_BUSINESS
    assert result.parameters["niche"] == "pet toys"

@pytest.mark.asyncio
async def test_classify_intent_query(master_ai, mock_llm_router):
    """Test that data requests are classified as queries."""
    mock_llm_router.complete = AsyncMock(return_value='{"type": "query", "action": null, "parameters": {}, "confidence": 0.9}')
    
    result = await master_ai._classify_intent("How much money did we make?", "context")
    assert result.type == IntentType.QUERY

# --- Evolution Safe-guards ---

@pytest.mark.asyncio
async def test_evolution_rate_limiting(master_ai):
    """Test that evolution proposals are strictly rate-limited."""
    # Simulate that we've already hit the limit
    master_ai._evolution_count_this_hour = 100 
    
    # Mock the evolution engine so we can see if it gets called
    master_ai.evolution.propose_improvement = AsyncMock()
    
    await master_ai._consider_evolution("context")
    
    # Needs to verify it was NOT called
    master_ai.evolution.propose_improvement.assert_not_called()

@pytest.mark.asyncio
async def test_evolution_under_limit(master_ai):
    """Test that evolution is allowed when under the limit."""
    master_ai._evolution_count_this_hour = 0
    master_ai.evolution.propose_improvement = AsyncMock(return_value=None)
    
    await master_ai._consider_evolution("context")
    
    master_ai.evolution.propose_improvement.assert_called_once()
