"""
Simple Agent Example

Demonstrates basic agent concepts using the text_summarize skill handler directly.
This is a simplified standalone example that shows core concepts without needing
the full framework running.
"""

import asyncio
import sys
from pathlib import Path

# Add skills directory to path
repo_root = Path(__file__).parent.parent.parent
skills_path = repo_root / "code-exec" / "skills" / "text_summarize"
sys.path.insert(0, str(skills_path.parent))


async def main():
    """Run a simple agent that uses the text_summarize skill."""
    print("=" * 60)
    print("Simple Agent Example")
    print("=" * 60)

    print("\n📝 Task: Summarize a research article about AI agents")
    print("\n🤖 Agent Configuration:")
    print("   - Role: Research Assistant")
    print("   - Skill: text_summarize")
    print("   - Execution: Direct handler invocation")

    # Sample text to summarize
    sample_text = """
    Multi-agent systems represent a significant advancement in artificial intelligence,
    enabling multiple specialized agents to collaborate on complex tasks. These systems
    leverage the strengths of individual agents while compensating for their weaknesses
    through coordination and communication protocols.

    Recent research has shown that multi-agent architectures outperform single-agent
    systems in domains requiring diverse expertise, parallel processing, and adaptive
    decision-making. Key benefits include improved scalability, fault tolerance, and
    the ability to decompose complex problems into manageable subtasks.

    Challenges remain in areas such as agent coordination, conflict resolution, and
    ensuring coherent outputs when multiple agents contribute to a single solution.
    Enterprise adoption is growing, particularly in sectors like finance, healthcare,
    and autonomous systems where reliability and auditability are critical.
    """

    print("\n🔄 Execution Flow:")
    print("   [1/3] Loading text_summarize skill handler...")

    try:
        # Import the handler directly
        from text_summarize.handler import summarize

        print("   [2/3] Executing summarization...")

        # Execute the skill with different styles
        styles = ["concise", "bullet-points"]

        for style in styles:
            import time
            start_time = time.time()

            result = summarize(
                text=sample_text.strip(),
                style=style,
                max_sentences=3
            )

            exec_time_ms = int((time.time() - start_time) * 1000)

            print(f"\n✅ Summary ({style}):")
            print(f"   {result.get('summary', 'No summary generated')}")
            print(f"   Original length: {result.get('original_length', 0)} chars")
            print(f"   Summary length: {result.get('summary_length', 0)} chars")
            print(f"   Compression ratio: {result.get('compression_ratio', 0):.2f}")
            print(f"   Execution time: {exec_time_ms}ms")

        print("\n   [3/3] Task completed successfully!")

    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"   1. Make sure you're in the repository root or examples/01-simple-agent")
        print(f"   2. Verify text_summarize skill exists at: {skills_path}")
        print(f"   3. Check that handler.py is present in the skill directory")
        return
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\n💡 What you learned:")
    print("   ✓ How to load and execute a skill handler")
    print("   ✓ How to pass inputs to skills")
    print("   ✓ How to handle skill outputs")
    print("\nNext steps:")
    print("  - See examples/02-multi-step-workflow/ for multi-agent orchestration")
    print("  - See examples/03-custom-skill/ for creating your own skills")
    print("  - Read docs/ for comprehensive guides")


if __name__ == "__main__":
    asyncio.run(main())
