#!/usr/bin/env python3
"""
Test token tracking and streaming functionality.
"""

import os
import sys
from pathlib import Path

# Add kautilya to path
sys.path.insert(0, str(Path(__file__).parent))

def test_streaming_with_tokens():
    """Test that streaming works and returns token usage."""
    from kautilya.llm_client import KautilyaLLMClient
    from kautilya.tool_executor import ToolExecutor

    # Initialize
    client = KautilyaLLMClient()
    executor = ToolExecutor(config_dir=".kautilya")

    print("=" * 60)
    print("Testing Streaming and Token Tracking")
    print("=" * 60)

    # Test query
    query = "What is 2 + 2? Answer in one short sentence."

    print(f"\nQuery: {query}\n")
    print("Streaming response:")
    print("-" * 60)

    collected_chunks = []
    result = None

    # Capture chunks and return value
    chat_gen = client.chat(query, tool_executor=executor, stream=True)

    while True:
        try:
            chunk = next(chat_gen)
            # Skip iteration markers and tool execution markers
            if chunk and not chunk.startswith("\n\n[Iteration") and not chunk.startswith("\n\n> Executing"):
                print(chunk, end="", flush=True)
                collected_chunks.append(chunk)
        except StopIteration as e:
            result = e.value
            break

    print("\n" + "-" * 60)

    # Check results
    print(f"\nChunks received: {len(collected_chunks)}")
    print(f"Total length: {len(''.join(collected_chunks))} chars")

    if result and isinstance(result, dict):
        usage = result.get("usage")
        if usage:
            print("\n✅ Token Usage Captured:")
            print(f"  Input Tokens:  {usage['prompt_tokens']:,}")
            print(f"  Output Tokens: {usage['completion_tokens']:,}")
            print(f"  Total Tokens:  {usage['total_tokens']:,}")
        else:
            print("\n❌ No token usage in result")
            print(f"Result: {result}")
    else:
        print(f"\n❌ No result captured (got: {type(result)})")

    # Verify streaming worked
    if len(collected_chunks) > 1:
        print("\n✅ Streaming is working (received multiple chunks)")
    elif len(collected_chunks) == 1:
        print("\n⚠️  Only 1 chunk received (streaming may not be working properly)")
    else:
        print("\n❌ No chunks received")

    print("\n" + "=" * 60)

    if collected_chunks and result and isinstance(result, dict) and result.get("usage"):
        print("✅ ALL TESTS PASSED")
        print("\n1. ✓ Streaming is working")
        print("2. ✓ Token tracking is working")
        print("3. ✓ Usage info returned correctly")
        return 0
    else:
        print("❌ TESTS FAILED")
        if not collected_chunks:
            print("\n- Streaming not working (no chunks)")
        if not result or not isinstance(result, dict):
            print(f"\n- Return value not dict (got {type(result)})")
        if not (result and result.get("usage")):
            print("\n- Token usage not captured")
        return 1

def main():
    """Run tests."""
    try:
        return test_streaming_with_tokens()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
