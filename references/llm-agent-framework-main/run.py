#!/usr/bin/env python3
"""
CLI Tool for LLM Agent Framework

Usage:
    python run.py single "Task description"
    python run.py multi "Complex task requiring multiple agents"
    python run.py interactive
    python run.py demo
"""

import argparse
import sys

from agent_framework.agents.supervisor import SupervisorAgent
from agent_framework.agents.react_agent import ReActAgent
from agent_framework.agents.base_agent import BaseAgent


class MockLLM:
    """Mock LLM for testing without API keys"""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate mock response"""
        self.call_count += 1

        # Simple pattern matching
        if "research" in prompt.lower():
            return "FINAL ANSWER: Based on research, I found relevant information about the topic."
        elif "code" in prompt.lower() or "python" in prompt.lower():
            return "FINAL ANSWER: Here's the code solution:\n\ndef solution():\n    return 'Hello World'"
        elif "analyze" in prompt.lower():
            return "FINAL ANSWER: Analysis complete. Key findings show positive trends."
        else:
            return "FINAL ANSWER: Task completed successfully."


def create_mock_tools():
    """Create mock tools for testing"""
    from agent_framework.tools.base_tool import BaseTool

    class SearchTool(BaseTool):
        name = "search"
        description = "Search for information"

        def run(self, query: str) -> str:
            return f"Found information about: {query}"

    class CalculatorTool(BaseTool):
        name = "calculator"
        description = "Perform calculations"

        def run(self, expression: str) -> str:
            try:
                result = eval(expression)
                return f"Result: {result}"
            except:
                return "Error: Invalid expression"

    class CodeTool(BaseTool):
        name = "code_executor"
        description = "Execute Python code"

        def run(self, code: str) -> str:
            try:
                exec_result = eval(code)
                return f"Executed: {exec_result}"
            except:
                return "Error: Code execution failed"

    return {
        "search": SearchTool(),
        "calculator": CalculatorTool(),
        "code_executor": CodeTool()
    }


def single_agent_command(args):
    """Run single ReAct agent"""
    print("🤖 Initializing Single Agent Mode")
    print("=" * 60)

    # Create mock LLM
    llm = MockLLM()
    tools = create_mock_tools()

    # Create agent
    agent = ReActAgent(
        name="assistant",
        description="Helpful AI assistant",
        llm=llm,
        tools=tools,
        max_iterations=args.max_iterations
    )

    print(f"\n📝 Task: {args.task}")
    print(f"🔧 Available Tools: {', '.join(tools.keys())}")
    print("\n⏳ Running agent...\n")

    # Run agent
    result = agent.run(args.task)

    print("\n" + "=" * 60)
    print("✅ Result")
    print("=" * 60)
    print(f"\n{result}")
    print(f"\n📊 LLM Calls: {llm.call_count}")
    print(f"💾 Memory Entries: {len(agent.memory)}")


def multi_agent_command(args):
    """Run multi-agent system"""
    print("🤖 Initializing Multi-Agent System")
    print("=" * 60)

    # Create mock LLM
    llm = MockLLM()
    tools = create_mock_tools()

    # Create supervisor
    supervisor = SupervisorAgent(llm=llm, max_iterations=args.max_iterations)

    # Create specialized agents
    research_agent = ReActAgent(
        name="researcher",
        description="Researches information and gathers data",
        llm=llm,
        tools=tools
    )

    code_agent = ReActAgent(
        name="coder",
        description="Writes and reviews code",
        llm=llm,
        tools=tools
    )

    analyst_agent = ReActAgent(
        name="analyst",
        description="Analyzes data and provides insights",
        llm=llm,
        tools=tools
    )

    # Register agents
    supervisor.register_agent(research_agent)
    supervisor.register_agent(code_agent)
    supervisor.register_agent(analyst_agent)

    print(f"\n📝 Task: {args.task}")
    print(f"👥 Agents: {', '.join(supervisor.list_agents())}")
    print("\n⏳ Running multi-agent system...\n")

    # Run supervisor
    result = supervisor.run(args.task)

    print("\n" + "=" * 60)
    print("✅ Final Result")
    print("=" * 60)

    if result.get('success'):
        print(f"\n💡 Answer: {result['answer']}")
        print(f"\n📊 Iterations: {result.get('iterations', 'N/A')}")
        print(f"📋 Results from {len(result.get('results', []))} agent operations")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")


def interactive_command(args):
    """Interactive agent mode"""
    print("🤖 Interactive Agent Mode")
    print("=" * 60)
    print("Commands: single, multi, tools, quit\n")

    llm = MockLLM()
    tools = create_mock_tools()

    # Create agents
    single_agent = ReActAgent(
        name="assistant",
        description="Helpful assistant",
        llm=llm,
        tools=tools
    )

    supervisor = SupervisorAgent(llm=llm)
    for name in ["researcher", "coder", "analyst"]:
        agent = ReActAgent(
            name=name,
            description=f"{name} agent",
            llm=llm,
            tools=tools
        )
        supervisor.register_agent(agent)

    while True:
        try:
            cmd = input("🔧 Enter mode (single/multi/tools/quit): ").strip().lower()

            if cmd in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            elif cmd == 'single':
                task = input("📝 Enter task: ")
                result = single_agent.run(task)
                print(f"\n✅ {result}\n")

            elif cmd == 'multi':
                task = input("📝 Enter task: ")
                result = supervisor.run(task)
                if result.get('success'):
                    print(f"\n✅ {result['answer']}\n")
                else:
                    print(f"\n❌ {result.get('error')}\n")

            elif cmd == 'tools':
                print("\n🔧 Available Tools:")
                for name, tool in tools.items():
                    print(f"   • {name}: {tool.description}")

            else:
                print("⚠️  Unknown command")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def demo_command(args):
    """Run demonstration"""
    print("\n" + "=" * 60)
    print("🎭 Agent Framework Demo")
    print("=" * 60)

    llm = MockLLM()
    tools = create_mock_tools()

    # Demo 1: Single Agent
    print("\n📌 Demo 1: Single ReAct Agent")
    print("-" * 60)

    agent = ReActAgent(
        name="demo_agent",
        description="Demo agent",
        llm=llm,
        tools=tools
    )

    task = "What is 2 + 2?"
    print(f"Task: {task}")
    result = agent.run(task)
    print(f"Result: {result}")

    # Demo 2: Multi-Agent
    print("\n📌 Demo 2: Multi-Agent System")
    print("-" * 60)

    supervisor = SupervisorAgent(llm=llm)

    researcher = ReActAgent(name="researcher", description="Researcher", llm=llm, tools=tools)
    coder = ReActAgent(name="coder", description="Coder", llm=llm, tools=tools)

    supervisor.register_agent(researcher)
    supervisor.register_agent(coder)

    task = "Research AI trends and create code to visualize them"
    print(f"Task: {task}")

    result = supervisor.run(task)
    if result.get('success'):
        print(f"Result: {result['answer'][:100]}...")

    # Demo 3: Tool Usage
    print("\n📌 Demo 3: Tool Execution")
    print("-" * 60)

    for tool_name, tool in tools.items():
        print(f"\n• {tool_name}: {tool.description}")
        if tool_name == "calculator":
            print(f"  Test: 2 + 2 = {tool.run('2+2')}")
        elif tool_name == "search":
            print(f"  Test: {tool.run('Python')}")

    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Agent Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py single "Calculate 2 + 2"
  python run.py multi "Research AI and write code"
  python run.py demo
  python run.py interactive
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single agent command
    single_parser = subparsers.add_parser('single', help='Run single agent')
    single_parser.add_argument('task', help='Task for the agent')
    single_parser.add_argument('--max-iterations', type=int, default=10,
                              help='Maximum iterations')

    # Multi-agent command
    multi_parser = subparsers.add_parser('multi', help='Run multi-agent system')
    multi_parser.add_argument('task', help='Complex task')
    multi_parser.add_argument('--max-iterations', type=int, default=10,
                              help='Maximum iterations')

    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')

    # Demo command
    subparsers.add_parser('demo', help='Run demonstration')

    args = parser.parse_args()

    if args.command == 'single':
        single_agent_command(args)
    elif args.command == 'multi':
        multi_agent_command(args)
    elif args.command == 'interactive':
        interactive_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
