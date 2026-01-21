#!/usr/bin/env python3
"""
Test script for the diary and reflection memory system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.reflective_loop import ReflectiveAgentLoop
from src.memory.manager import MemoryManager


async def test_diary_system():
    """Test the diary and reflection system."""

    print("🧠 Testing King AI Memory System")
    print("=" * 50)

    try:
        # Initialize memory manager
        print("📚 Initializing memory manager...")
        memory_manager = MemoryManager()
        print("✅ Memory manager initialized")

        # Initialize reflective agent loop with diary
        print("🤖 Initializing reflective agent loop...")
        agent_loop = ReflectiveAgentLoop(
            project_id="test_project",
            memory_manager=memory_manager,
            max_iterations=2,
            min_quality_score=0.5
        )
        print("✅ Agent loop initialized")

        # Test objective
        objective = "Create a simple Python function that adds two numbers and returns the result"

        print(f"🎯 Objective: {objective}")
        print("\n🤖 Running reflective agent loop...")

        # Run the agent loop
        result = await agent_loop.run(
            objective=objective,
            context={"language": "python", "requirements": ["simple", "well-documented"]},
            success_criteria=[
                "Function is implemented correctly",
                "Code follows Python conventions",
                "Function has proper documentation"
            ]
        )

        print("\n📊 Results:")
        print(f"  Success: {result.success}")
        print(f"  Iterations: {result.iteration_count}")
        print(f"  Duration: {result.duration_ms}ms")

        if result.final_output:
            print(f"\n💻 Generated Code:\n{result.final_output}")

        print("\n📝 Checking for diary entries...")        # Check if diary was created
        diary_dir = Path(".king_ai/memory/diary")
        if diary_dir.exists():
            entries = list(diary_dir.glob("*.md"))
            print(f"  Found {len(entries)} diary entries:")
            for entry in entries:
                print(f"    - {entry.name}")
                # Show first few lines
                content = entry.read_text()
                lines = content.split('\n')[:10]
                print(f"      Preview: {lines[0]}")
        else:
            print("  No diary directory found")

        print("\n🧠 Checking for memory updates...")        # Check long-term memory
        try:
            memories = await memory_manager.search_memories("test_project", "test", limit=5)
            print(f"  Found {len(memories)} long-term memories")
        except Exception as e:
            print(f"  Error checking memories: {e}")

        print("\n✅ Memory system test complete!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting memory system test...")
    asyncio.run(test_diary_system())