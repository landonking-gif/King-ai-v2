"""
Unit tests for ReAct Agent
"""

import pytest
from unittest.mock import Mock, MagicMock
from agent_framework.agents.react_agent import ReActAgent
from agent_framework.agents.base_agent import AgentAction, AgentFinish


@pytest.fixture
def mock_llm():
    """Mock LLM"""
    llm = Mock()
    llm.generate = Mock(return_value="Thought: I need to search\nFinal Answer: Test answer")
    return llm


@pytest.fixture
def mock_tools():
    """Mock tools"""
    search_tool = Mock()
    search_tool.name = "search"
    search_tool.description = "Search the web"
    search_tool.run = Mock(return_value="Search results")

    calculator_tool = Mock()
    calculator_tool.name = "calculator"
    calculator_tool.description = "Calculate math expressions"
    calculator_tool.run = Mock(return_value="42")

    tools = {
        "search": search_tool,
        "calculator": calculator_tool
    }
    return tools


@pytest.fixture
def react_agent(mock_llm, mock_tools):
    """Create ReAct agent for testing"""
    agent = ReActAgent(
        name="test_agent",
        description="Test ReAct agent",
        llm=mock_llm,
        tools=mock_tools,
        max_iterations=5
    )
    return agent


def test_react_agent_initialization(react_agent):
    """Test agent initialization"""
    assert react_agent.name == "test_agent"
    assert react_agent.description == "Test ReAct agent"
    assert react_agent.max_iterations == 5
    assert len(react_agent.tools) == 2


def test_prompt_building(react_agent):
    """Test prompt building"""
    prompt = react_agent._build_prompt("What is 2 + 2?")

    assert "What is 2 + 2?" in prompt
    assert "search" in prompt
    assert "calculator" in prompt
    assert "Thought:" in prompt
    assert "Action:" in prompt
    assert "Final Answer:" in prompt


def test_parse_final_answer():
    """Test parsing final answer"""
    agent = ReActAgent(
        name="test",
        description="test",
        llm=Mock(),
        tools={}
    )

    output = agent._parse_output("Thought: I know this\nFinal Answer: 42")

    assert isinstance(output, AgentFinish)
    assert output.output == "42"


def test_parse_action():
    """Test parsing action"""
    agent = ReActAgent(
        name="test",
        description="test",
        llm=Mock(),
        tools={}
    )

    output = agent._parse_output("Thought: I need to search\nAction: search\nAction Input: test query")

    assert isinstance(output, AgentAction)
    assert output.tool == "search"
    assert output.tool_input == {"query": "test query"}


def test_execute_action(react_agent):
    """Test executing an action"""
    action = AgentAction(
        tool="search",
        tool_input={"query": "test"},
        log="test"
    )

    result = react_agent.execute(action)

    assert result == "Search results"


def test_run_with_final_answer(react_agent):
    """Test running agent with direct final answer"""
    react_agent.llm.generate = Mock(return_value="Final Answer: The answer is 42")

    result = react_agent.run("What is the meaning of life?")

    assert result == "The answer is 42"


def test_max_iterations(react_agent):
    """Test max iterations limit"""
    # Make LLM return thoughts but never final answer
    react_agent.llm.generate = Mock(return_value="Thought: I need to think more\nAction: search\nAction Input: test")

    result = react_agent.run("Test question")

    assert "maximum iterations" in result.lower()


def test_tool_execution_error(react_agent):
    """Test handling tool execution errors"""
    # Make tool raise error
    react_agent.tools["search"].run = Mock(side_effect=Exception("Tool failed"))

    action = AgentAction(
        tool="search",
        tool_input={"query": "test"},
        log="test"
    )

    result = react_agent.execute(action)

    assert "Error executing tool" in result


def test_memory_storage(react_agent):
    """Test that agent stores interactions in memory"""
    react_agent.llm.generate = Mock(return_value="Final Answer: Test")

    react_agent.run("Test question")

    assert len(react_agent.memory) > 0
    assert "iteration" in react_agent.memory[0]
