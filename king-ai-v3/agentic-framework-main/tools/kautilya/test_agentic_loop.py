#!/usr/bin/env python3
"""
Test script for agentic loop functionality.

This script tests that the agent can perform multiple rounds of tool calls
to complete complex multi-step tasks.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kautilya.llm_client import KautilyaLLMClient
from kautilya.tool_executor import ToolExecutor


def test_single_step_query():
    """Test simple single-step query (baseline)."""
    print("=" * 80)
    print("TEST 1: Single-step query (file_glob)")
    print("=" * 80)

    client = KautilyaLLMClient()
    executor = ToolExecutor()

    query = "Find all Python files in the current directory using file_glob"
    print(f"\nQuery: {query}\n")

    result = None
    for chunk in client.chat(query, tool_executor=executor, stream=True):
        print(chunk, end="", flush=True)
        result = chunk

    print("\n")
    return result


def test_multi_step_query():
    """Test complex multi-step query requiring multiple tool iterations."""
    print("=" * 80)
    print("TEST 2: Multi-step query (glob -> read -> summarize)")
    print("=" * 80)

    client = KautilyaLLMClient()
    executor = ToolExecutor()

    query = """Find Python files in the kautilya directory, then read the llm_client.py file
    and tell me about the chat() method implementation."""

    print(f"\nQuery: {query}\n")

    result = None
    for chunk in client.chat(query, tool_executor=executor, stream=True, max_tool_iterations=5):
        print(chunk, end="", flush=True)
        result = chunk

    print("\n")
    return result


def test_iterative_refinement():
    """Test iterative refinement with multiple iterations."""
    print("=" * 80)
    print("TEST 3: Iterative refinement (search -> read -> analyze)")
    print("=" * 80)

    client = KautilyaLLMClient()
    executor = ToolExecutor()

    query = """Search for 'agentic loop' in Python files, then read one of the matching files
    and explain what you found."""

    print(f"\nQuery: {query}\n")

    result = None
    for chunk in client.chat(query, tool_executor=executor, stream=True, max_tool_iterations=5):
        print(chunk, end="", flush=True)
        result = chunk

    print("\n")
    return result


def test_max_iterations_limit():
    """Test that max_tool_iterations is respected."""
    print("=" * 80)
    print("TEST 4: Max iterations limit (should stop at 2)")
    print("=" * 80)

    client = KautilyaLLMClient()
    executor = ToolExecutor()

    query = "Find all YAML files and read each one"

    print(f"\nQuery: {query}\n")
    print("Setting max_tool_iterations=2\n")

    result = None
    for chunk in client.chat(query, tool_executor=executor, stream=True, max_tool_iterations=2):
        print(chunk, end="", flush=True)
        result = chunk

    print("\n")
    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AGENTIC LOOP TEST SUITE")
    print("=" * 80 + "\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it before running tests:")
        print("  export OPENAI_API_KEY=your-key-here")
        return 1

    try:
        # Run tests
        test_single_step_query()
        print("\n" + "=" * 80 + "\n")

        test_multi_step_query()
        print("\n" + "=" * 80 + "\n")

        test_iterative_refinement()
        print("\n" + "=" * 80 + "\n")

        test_max_iterations_limit()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
