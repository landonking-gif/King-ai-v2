"""
Basic Usage Example for LLM Agent Framework

Demonstrates how to create and use agents for various tasks.
"""

from agent_framework.agents.react_agent import ReActAgent
from agent_framework.agents.supervisor import SupervisorAgent
from agent_framework.tools.base_tool import SearchTool, CalculatorTool, PythonREPLTool
from agent_framework.tools.web_tools import WebSearchTool, WikipediaTool
from agent_framework.tools.code_tools import PythonExecutorTool, CodeAnalyzerTool
from agent_framework.memory.conversation_memory import ConversationMemory


class SimpleLLM:
    """Simple LLM wrapper for demonstration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        except ImportError:
            # Fallback for demo without OpenAI
            return f"[Demo Response] Processing: {prompt[:100]}..."
        except Exception as e:
            return f"Error: {str(e)}"


def example_react_agent():
    """Example: Using a ReAct agent for research"""
    print("\n" + "=" * 50)
    print("Example 1: ReAct Agent for Research")
    print("=" * 50)
    
    # Initialize LLM
    llm = SimpleLLM()
    
    # Create tools
    tools = [
        SearchTool(),
        WikipediaTool(),
        CalculatorTool()
    ]
    
    # Create agent
    agent = ReActAgent(
        name="ResearchAgent",
        description="An agent that researches topics using web search and Wikipedia",
        llm=llm,
        tools=tools,
        max_iterations=5
    )
    
    # Run agent
    task = "What is the population of Tokyo and how does it compare to New York?"
    print(f"\nTask: {task}")
    print("-" * 40)
    
    result = agent.run(task)
    print(f"\nResult: {result}")


def example_code_agent():
    """Example: Using an agent for code tasks"""
    print("\n" + "=" * 50)
    print("Example 2: Code Agent")
    print("=" * 50)
    
    llm = SimpleLLM()
    
    tools = [
        PythonExecutorTool(),
        CodeAnalyzerTool(),
        CalculatorTool()
    ]
    
    agent = ReActAgent(
        name="CodeAgent",
        description="An agent that writes and executes Python code",
        llm=llm,
        tools=tools,
        max_iterations=5
    )
    
    task = "Calculate the first 10 Fibonacci numbers"
    print(f"\nTask: {task}")
    print("-" * 40)
    
    result = agent.run(task)
    print(f"\nResult: {result}")


def example_multi_agent():
    """Example: Using supervisor with multiple agents"""
    print("\n" + "=" * 50)
    print("Example 3: Multi-Agent Orchestration")
    print("=" * 50)
    
    llm = SimpleLLM()
    
    # Create specialized agents
    research_agent = ReActAgent(
        name="Researcher",
        description="Researches information from the web",
        llm=llm,
        tools=[SearchTool(), WikipediaTool()]
    )
    
    code_agent = ReActAgent(
        name="Coder",
        description="Writes and executes Python code",
        llm=llm,
        tools=[PythonExecutorTool(), CalculatorTool()]
    )
    
    # Create supervisor
    supervisor = SupervisorAgent(llm=llm, max_iterations=5)
    supervisor.register_agent(research_agent)
    supervisor.register_agent(code_agent)
    
    # Run complex task
    task = "Research the GDP of Japan and create a Python script to visualize it"
    print(f"\nTask: {task}")
    print("-" * 40)
    
    result = supervisor.run(task)
    
    print(f"\nSuccess: {result.get('success')}")
    print(f"Answer: {result.get('answer', 'N/A')[:500]}...")
    print(f"Iterations: {result.get('iterations', 'N/A')}")


def example_with_memory():
    """Example: Using agents with conversation memory"""
    print("\n" + "=" * 50)
    print("Example 4: Agent with Memory")
    print("=" * 50)
    
    llm = SimpleLLM()
    memory = ConversationMemory(max_messages=10)
    
    # Create agent
    agent = ReActAgent(
        name="MemoryAgent",
        description="An agent that remembers conversation context",
        llm=llm,
        tools=[SearchTool(), CalculatorTool()]
    )
    
    # Simulate conversation
    questions = [
        "What is Python?",
        "What are its main features?",
        "How does it compare to Java?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        memory.add_message("user", question)
        
        # Get context from memory
        context = memory.get_context_string()
        
        # Run agent with context
        result = agent.run(f"Context:\n{context}\n\nQuestion: {question}")
        
        print(f"Agent: {result[:200]}...")
        memory.add_message("assistant", result)
    
    print("\n--- Conversation History ---")
    print(memory.get_context_string())


def main():
    """Run all examples"""
    print("🤖 LLM Agent Framework - Examples")
    print("=" * 50)
    
    # Run examples
    example_react_agent()
    example_code_agent()
    example_multi_agent()
    example_with_memory()
    
    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
