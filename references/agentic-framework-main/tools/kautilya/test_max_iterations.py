#!/usr/bin/env python3
"""
Test that max iterations bug is fixed - output should be shown even when hitting max iterations.
"""

import os
import sys
from pathlib import Path

# Add kautilya to path
sys.path.insert(0, str(Path(__file__).parent))

def test_max_iterations_output():
    """Test that we get output when hitting max iterations."""
    from kautilya.llm_client import KautilyaLLMClient
    from kautilya.tool_executor import ToolExecutor

    # Initialize with max 2 iterations (low to hit limit quickly)
    client = KautilyaLLMClient(max_iterations=2)
    executor = ToolExecutor(config_dir=".kautilya")

    print("=" * 60)
    print("Testing Max Iterations Output")
    print("=" * 60)

    # This query should trigger multiple tool calls and hit the limit
    query = "List Python files, then count them, then read one of them"

    print(f"\nQuery: {query}")
    print(f"Max iterations: 2 (deliberately low to test limit)\n")
    print("Expected behavior:")
    print("  - Iteration 1: Calls file_glob")
    print("  - Iteration 2: Calls another tool (file_grep or file_read)")
    print("  - Max iterations reached")
    print("  - ✅ SHOULD STILL SHOW FINAL RESPONSE\n")
    print("-" * 60)

    collected_chunks = []
    result = None

    # Capture chunks and return value
    chat_gen = client.chat(query, tool_executor=executor, stream=True)

    iteration_count = 0
    tool_calls = []
    got_max_iterations_msg = False
    got_final_response = False

    while True:
        try:
            chunk = next(chat_gen)

            # Track iterations
            if "[Iteration " in chunk:
                iteration_count += 1

            # Track tool calls
            if "> Executing: " in chunk:
                import re
                match = re.search(r'> Executing: ([a-z_]+)', chunk)
                if match:
                    tool_calls.append(match.group(1))

            # Check for max iterations message
            if "[Max iterations reached" in chunk:
                got_max_iterations_msg = True
                print("\n✓ Detected max iterations message")

            # Skip markers
            if (chunk.startswith("\n\n[Iteration") or
                chunk.startswith("\n\n> Executing") or
                "[Max iterations reached" in chunk):
                continue

            # Collect actual content
            if chunk.strip():
                collected_chunks.append(chunk)
                print(chunk, end="", flush=True)
                if not got_max_iterations_msg:
                    # This is response before hitting limit (shouldn't happen with our query)
                    pass
                else:
                    # This is the final response AFTER hitting limit - this is what we're testing for!
                    got_final_response = True

        except StopIteration as e:
            result = e.value
            break

    print("\n" + "-" * 60)

    # Verify results
    print(f"\nIterations executed: {iteration_count}")
    print(f"Tools called: {len(tool_calls)} - {tool_calls}")
    print(f"Hit max iterations: {got_max_iterations_msg}")
    print(f"Got final response: {got_final_response}")
    print(f"Total response chunks: {len(collected_chunks)}")
    print(f"Total response length: {len(''.join(collected_chunks))} chars")

    print("\n" + "=" * 60)

    # Check if bug is fixed
    if got_max_iterations_msg and got_final_response:
        print("✅ BUG FIXED!")
        print("\n✓ Max iterations was reached")
        print("✓ Final response was generated and shown")
        print("✓ User can see the output")
        return 0
    elif got_max_iterations_msg and not got_final_response:
        print("❌ BUG STILL EXISTS!")
        print("\n✓ Max iterations was reached")
        print("✗ No final response shown")
        print("✗ User sees no output (THIS IS THE BUG)")
        return 1
    else:
        print("⚠️  Test inconclusive")
        print(f"\nMax iterations msg: {got_max_iterations_msg}")
        print(f"Final response: {got_final_response}")
        print("Query may have completed before hitting max iterations")
        print("Try running again or with more complex query")
        return 2

def main():
    """Run test."""
    try:
        return test_max_iterations_output()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
