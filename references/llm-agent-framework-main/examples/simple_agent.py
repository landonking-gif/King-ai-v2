"""
Simple Agent Example

Demonstrates basic usage of the ReAct agent.
"""

from agent_framework.agents.react_agent import ReActAgent
from agent_framework.tools.base_tool import SearchTool, CalculatorTool


class SimpleLLM:
    """Mock LLM for demonstration"""
    
    def generate(self, prompt: str) -> str:
        """Generate mock response"""
        return """Thought: I need to search for information about AI
Action: search
Action Input: artificial intelligence latest developments
"""


def main():
    # Create tools
    tools = [
        SearchTool(),
        CalculatorTool()
    ]
    
    # Create LLM
    llm = SimpleLLM()
    
    # Create agent
    agent = ReActAgent(
        name="ResearchAgent",
        description="An agent that can search and calculate",
        llm=llm,
        tools=tools,
        max_iterations=5
    )
    
    # Run agent
    result = agent.run("What is 2 + 2?")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
