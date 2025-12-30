"""
Unit tests for the Agent System.
Tests the Agent Router and individual Agent execution logic.
"""

import pytest
from src.agents.router import AgentRouter
from src.agents.research import ResearchAgent
from unittest.mock import AsyncMock, patch

# --- Agent Router Tests ---

@pytest.mark.asyncio
async def test_router_executes_correct_agent():
    router = AgentRouter()
    
    # Mock the specific agent to avoid actual LLM calls
    mock_agent = AsyncMock()
    mock_agent.execute.return_value = {"success": True, "output": "mocked"}
    router.agents["research"] = mock_agent
    
    task = {"agent": "research", "input": {"query": "test"}}
    result = await router.execute(task)
    
    assert result["success"] == True
    mock_agent.execute.assert_called_once_with(task)

@pytest.mark.asyncio
async def test_router_handles_unknown_agent():
    router = AgentRouter()
    task = {"agent": "fake_agent"}
    
    result = await router.execute(task)
    
    assert result["success"] == False
    assert "Unknown agent" in result["error"]

@pytest.mark.asyncio
async def test_router_handles_missing_agent_field():
    router = AgentRouter()
    task = {"input": "foo"} # Missing 'agent' key
    
    result = await router.execute(task)
    
    assert result["success"] == False
    assert "No agent specified" in result["error"]

# --- Research Agent Tests ---

@pytest.mark.asyncio
async def test_research_agent_web_search():
    agent = ResearchAgent()
    
    # Mock the internal LLM call to avoid network
    agent._ask_llm = AsyncMock(return_value="Synthesized summary of search")
    
    task = {
        "input": {
            "type": "web_search",
            "query": "latest AI trends"
        }
    }
    
    result = await agent.execute(task)
    
    assert result["success"] == True
    assert result["output"]["query"] == "latest AI trends"
    assert result["output"]["summary"] == "Synthesized summary of search"
    agent._ask_llm.assert_called_once()

@pytest.mark.asyncio
async def test_research_agent_unknown_task_type():
    agent = ResearchAgent()
    task = {"input": {"type": "unknown_type"}}
    
    result = await agent.execute(task)
    
    assert result["success"] == False
    assert "Unknown research type" in result["error"]
