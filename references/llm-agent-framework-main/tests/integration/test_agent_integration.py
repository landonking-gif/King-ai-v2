"""
Integration tests for LLM Agent Framework

These tests verify multi-agent orchestration and tool integration.
"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.generate = Mock(side_effect=lambda prompt: "Final Answer: Task completed successfully.")
    return llm


@pytest.fixture
def mock_tools():
    """Mock tools for testing"""
    search_tool = Mock()
    search_tool.name = "search"
    search_tool.description = "Search the web for information"
    search_tool.run = Mock(return_value="Found relevant information")

    calculator = Mock()
    calculator.name = "calculator"
    calculator.description = "Perform mathematical calculations"
    calculator.run = Mock(return_value="42")

    code_tool = Mock()
    code_tool.name = "code_executor"
    code_tool.description = "Execute Python code"
    code_tool.run = Mock(return_value="Code executed successfully")

    return {
        "search": search_tool,
        "calculator": calculator,
        "code_executor": code_tool
    }


@pytest.mark.integration
def test_supervisor_with_multiple_agents(mock_llm, mock_tools):
    """Test supervisor coordinating multiple agents"""
    from agent_framework.agents.supervisor import SupervisorAgent
    from agent_framework.agents.react_agent import ReActAgent

    # Create supervisor
    supervisor = SupervisorAgent(llm=mock_llm, max_iterations=5)

    # Create specialized agents
    research_agent = ReActAgent(
        name="researcher",
        description="Researches information",
        llm=mock_llm,
        tools=mock_tools,
        max_iterations=3
    )

    code_agent = ReActAgent(
        name="coder",
        description="Writes code",
        llm=mock_llm,
        tools=mock_tools,
        max_iterations=3
    )

    # Register agents
    supervisor.register_agent(research_agent)
    supervisor.register_agent(code_agent)

    # Verify registration
    assert len(supervisor.list_agents()) == 2
    assert "researcher" in supervisor.list_agents()
    assert "coder" in supervisor.list_agents()


@pytest.mark.integration
def test_multi_agent_task_decomposition(mock_llm, mock_tools):
    """Test task decomposition across multiple agents"""
    from agent_framework.agents.supervisor import SupervisorAgent
    from agent_framework.agents.react_agent import ReActAgent

    supervisor = SupervisorAgent(llm=mock_llm)

    # Register agents
    research_agent = ReActAgent(
        name="researcher",
        description="Research agent",
        llm=mock_llm,
        tools=mock_tools
    )

    supervisor.register_agent(research_agent)

    # Test task planning
    planning_prompt = supervisor._build_planning_prompt("Research AI trends")
    assert "Research AI trends" in planning_prompt
    assert "researcher" in planning_prompt


@pytest.mark.integration
def test_agent_memory_and_context():
    """Test that agents maintain memory and context"""
    from agent_framework.agents.react_agent import ReActAgent
    from agent_framework.agents.base_agent import BaseAgent

    llm = Mock()
    llm.generate = Mock(return_value="Final Answer: Done")

    tools = {"test": Mock(name="test", description="Test tool", run=Mock(return_value="result"))}

    agent = ReActAgent(
        name="test_agent",
        description="Test",
        llm=llm,
        tools=tools
    )

    # Run task
    agent.run("Test task")

    # Check memory
    assert len(agent.memory) > 0
    assert any('iteration' in m for m in agent.memory)


@pytest.mark.integration
def test_tool_execution_chain():
    """Test sequential tool execution"""
    from agent_framework.agents.react_agent import ReActAgent
    from agent_framework.agents.base_agent import AgentAction

    llm = Mock()

    # Simulate tool usage sequence
    llm.generate = Mock(
        side_effect=[
            "Thought: I need to search\nAction: search\nAction Input: AI trends",
            "Thought: I have the info\nFinal Answer: AI is growing rapidly"
        ]
    )

    search_tool = Mock()
    search_tool.name = "search"
    search_tool.description = "Search tool"
    search_tool.run = Mock(return_value="AI trends found")

    agent = ReActAgent(
        name="tester",
        description="Test agent",
        llm=llm,
        tools={"search": search_tool}
    )

    result = agent.run("What are AI trends?")

    # Verify tool was called
    search_tool.run.assert_called_once()
    assert "AI" in result or "growing" in result


@pytest.mark.integration
def test_agent_error_handling():
    """Test agent handles tool errors gracefully"""
    from agent_framework.agents.react_agent import ReActAgent

    llm = Mock()
    llm.generate = Mock(return_value="Final Answer: Handled error")

    # Tool that raises error
    error_tool = Mock()
    error_tool.name = "error_tool"
    error_tool.description = "Tool that fails"
    error_tool.run = Mock(side_effect=Exception("Tool failed"))

    agent = ReActAgent(
        name="error_tester",
        description="Test error handling",
        llm=llm,
        tools={"error_tool": error_tool}
    )

    action = Mock()
    action.tool = "error_tool"
    action.tool_input = {"query": "test"}
    action.log = "Test"

    result = agent.execute(action)

    # Should return error message
    assert "Error" in result


@pytest.mark.integration
def test_supervisor_result_synthesis():
    """Test supervisor synthesizes results from multiple agents"""
    from agent_framework.agents.supervisor import SupervisorAgent, TaskResult

    llm = Mock()
    llm.generate = Mock(return_value="Synthesized result from all agents")

    supervisor = SupervisorAgent(llm=llm)

    # Create mock results
    results = [
        TaskResult(
            agent_name="agent1",
            task="Task 1",
            result="Result from agent 1",
            success=True
        ),
        TaskResult(
            agent_name="agent2",
            task="Task 2",
            result="Result from agent 2",
            success=True
        )
    ]

    synthesized = supervisor._synthesize_results("Main task", results)

    assert "agent1" in synthesized
    assert "agent2" in synthesized


@pytest.mark.integration
def test_agent_parallel_execution():
    """Test agents can work on independent tasks in parallel"""
    from agent_framework.agents.supervisor import SupervisorAgent
    from agent_framework.agents.react_agent import ReActAgent

    llm = Mock()
    llm.generate = Mock(return_value="Final Answer: Complete")

    supervisor = SupervisorAgent(llm=llm, max_iterations=3)

    # Create multiple agents
    agents = []
    for i in range(3):
        agent = ReActAgent(
            name=f"agent_{i}",
            description=f"Agent {i}",
            llm=llm,
            tools={}
        )
        supervisor.register_agent(agent)
        agents.append(agent)

    assert len(supervisor.list_agents()) == 3


@pytest.mark.integration
def test_agent_task_dependencies():
    """Test task execution with dependencies"""
    from agent_framework.agents.supervisor import AgentTask

    task1 = AgentTask(
        agent_name="agent1",
        task="First task",
        priority=0,
        dependencies=[]
    )

    task2 = AgentTask(
        agent_name="agent2",
        task="Second task (depends on first)",
        priority=1,
        dependencies=["agent1"]
    )

    assert task1.priority == 0
    assert "agent1" in task2.dependencies


@pytest.mark.integration
def test_agent_tool_registration():
    """Test dynamic tool registration"""
    from agent_framework.agents.react_agent import ReActAgent

    llm = Mock()
    llm.generate = Mock(return_value="Final Answer: Done")

    agent = ReActAgent(
        name="tester",
        description="Test",
        llm=llm,
        tools={}
    )

    # Initially no tools
    assert len(agent.tools) == 0

    # Tool should be accessible through get_tool
    assert agent.get_tool("nonexistent") is None
