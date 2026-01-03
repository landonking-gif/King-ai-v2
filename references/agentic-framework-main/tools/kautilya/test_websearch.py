#!/usr/bin/env python3
"""
Test web search functionality with DuckDuckGo (default provider).
"""

import os
import sys
from pathlib import Path

# Add kautilya to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tool_executor_websearch():
    """Test that tool executor has web search methods."""
    from kautilya.tool_executor import ToolExecutor

    executor = ToolExecutor(config_dir=".kautilya")

    # Test 1: Check that web search methods exist
    assert hasattr(executor, '_exec_web_search'), "Missing _exec_web_search method"
    assert hasattr(executor, '_exec_web_search_duckduckgo'), "Missing _exec_web_search_duckduckgo method"
    assert hasattr(executor, '_exec_list_websearch_providers'), "Missing _exec_list_websearch_providers method"

    print("✓ Web search methods exist in ToolExecutor")

    # Test 2: Check default configuration
    config = executor._load_websearch_config()
    assert config['default_provider'] == 'duckduckgo', f"Default provider should be duckduckgo, got {config['default_provider']}"

    print(f"✓ Default provider is: {config['default_provider']}")

    # Test 3: List providers
    result = executor._exec_list_websearch_providers()
    assert result['success'], f"Failed to list providers: {result.get('error')}"
    assert result['default_provider'] == 'duckduckgo'

    print(f"✓ List providers works, found {len(result['providers'])} providers")

    # Test 4: Execute a real web search with DuckDuckGo
    print("\n🔍 Testing real DuckDuckGo web search...")
    result = executor._exec_web_search(
        query="Python programming language",
        max_results=3,
        provider="duckduckgo"
    )

    if not result['success']:
        print(f"❌ Web search failed: {result.get('error')}")
        return False

    print(f"✓ Web search successful!")
    print(f"  Provider: {result['provider']}")
    print(f"  Results: {result['result_count']}")

    if result['results']:
        print(f"\n  Sample result:")
        first = result['results'][0]
        print(f"    Title: {first['title'][:60]}...")
        print(f"    URL: {first['url']}")
        print(f"    Snippet: {first['snippet'][:100]}...")

    return True

def test_llm_client_tools():
    """Test that LLM client has web search tools registered."""
    from kautilya.llm_client import KAUTILYA_TOOLS, WEB_SEARCH_TOOLS

    # Test 1: Check WEB_SEARCH_TOOLS exists
    assert len(WEB_SEARCH_TOOLS) > 0, "WEB_SEARCH_TOOLS is empty"
    print(f"✓ WEB_SEARCH_TOOLS defined with {len(WEB_SEARCH_TOOLS)} tools")

    # Test 2: Check web_search tool is in KAUTILYA_TOOLS
    tool_names = [t['function']['name'] for t in KAUTILYA_TOOLS]
    assert 'web_search' in tool_names, "web_search tool not found in KAUTILYA_TOOLS"

    print(f"✓ web_search tool is registered")
    print(f"  Total tools available: {len(KAUTILYA_TOOLS)}")

    # Test 3: Verify web_search tool definition
    web_search_tool = next((t for t in KAUTILYA_TOOLS if t['function']['name'] == 'web_search'), None)
    assert web_search_tool is not None
    assert 'query' in web_search_tool['function']['parameters']['properties']

    print(f"✓ web_search tool properly defined with parameters")

    return True

def test_runtime_availability():
    """Test that web search is available at runtime."""
    from kautilya.tool_executor import ToolExecutor

    executor = ToolExecutor(config_dir=".kautilya")

    # Test executing via the generic execute method
    result = executor.execute('web_search', {
        'query': 'test query',
        'max_results': 1
    })

    # Should succeed or fail gracefully (not with "Unknown tool" error)
    assert 'Unknown tool' not in result.get('error', ''), f"web_search not recognized as a tool: {result}"

    print(f"✓ web_search tool is available via executor.execute()")

    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Web Search Integration")
    print("=" * 60)

    try:
        print("\n[1/3] Testing LLM Client Tools Registration...")
        if not test_llm_client_tools():
            print("❌ LLM client tools test failed")
            return 1

        print("\n[2/3] Testing Tool Executor Implementation...")
        if not test_tool_executor_websearch():
            print("❌ Tool executor test failed")
            return 1

        print("\n[3/3] Testing Runtime Availability...")
        if not test_runtime_availability():
            print("❌ Runtime availability test failed")
            return 1

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Web Search Ready!")
        print("=" * 60)
        print("\nDuckDuckGo web search is available by default.")
        print("No configuration needed - just ask questions that need web search!")
        print("\nExample queries:")
        print("  - What are the latest GPT models?")
        print("  - Current Bitcoin price")
        print("  - Python 3.13 new features")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
