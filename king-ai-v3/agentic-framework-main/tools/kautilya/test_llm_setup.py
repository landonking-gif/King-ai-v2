#!/usr/bin/env python3
"""
Test script to diagnose and fix LLM connection issues.

This script helps you:
1. Check if API key is set
2. Test the connection to OpenAI
3. Verify the LLM client works
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
parent_env = Path(__file__).parent.parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)
else:
    load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def check_api_key():
    """Check if OpenAI API key is set."""
    print("=" * 80)
    print("STEP 1: Checking API Key")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY is NOT set")
        print("\nTo fix this, run ONE of the following:\n")
        print("Option 1: Set for current session (temporary)")
        print("  export OPENAI_API_KEY='your-api-key-here'\n")
        print("Option 2: Set in .env file (recommended)")
        print("  echo 'OPENAI_API_KEY=your-api-key-here' > .env\n")
        print("Option 3: Add to your shell profile (permanent)")
        print("  echo 'export OPENAI_API_KEY=\"your-api-key-here\"' >> ~/.zshrc")
        print("  source ~/.zshrc\n")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return False

    print(f"✓ API key is set (length: {len(api_key)} chars)")

    # Check if key looks valid
    if not api_key.startswith("sk-"):
        print("⚠️  Warning: API key should start with 'sk-'")
        return False

    print("✓ API key format looks correct")
    return True


def test_openai_connection():
    """Test connection to OpenAI API."""
    print("\n" + "=" * 80)
    print("STEP 2: Testing OpenAI Connection")
    print("=" * 80)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("Making test API call...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello from Kautilya!'"}],
            max_tokens=20,
        )

        reply = response.choices[0].message.content
        print(f"✓ Connection successful!")
        print(f"✓ Response: {reply}")
        return True

    except ImportError:
        print("❌ OpenAI library not installed")
        print("\nTo fix this, run:")
        print("  pip install openai")
        return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")

        error_str = str(e).lower()
        if "api key" in error_str or "authentication" in error_str:
            print("\nThis looks like an API key issue. Verify:")
            print("1. Your API key is correct")
            print("2. You have API credits available")
            print("3. Visit: https://platform.openai.com/account/billing")
        elif "rate limit" in error_str:
            print("\nRate limit exceeded. Wait a moment and try again.")
        elif "timeout" in error_str or "connection" in error_str:
            print("\nNetwork connection issue. Check your internet connection.")

        return False


def test_llm_client():
    """Test Kautilya LLM client."""
    print("\n" + "=" * 80)
    print("STEP 3: Testing Kautilya LLM Client")
    print("=" * 80)

    try:
        from kautilya.llm_client import KautilyaLLMClient

        print("Initializing LLM client...")
        client = KautilyaLLMClient()

        print("✓ LLM client initialized")
        print(f"✓ Model: {client.model}")
        print(f"✓ History initialized: {len(client.history.messages)} messages")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're in the kautilya directory:")
        print("  cd tools/kautilya")
        return False

    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False


def test_tool_executor():
    """Test tool executor."""
    print("\n" + "=" * 80)
    print("STEP 4: Testing Tool Executor")
    print("=" * 80)

    try:
        from kautilya.tool_executor import ToolExecutor

        print("Initializing tool executor...")
        executor = ToolExecutor()

        print("Testing file_read tool...")
        result = executor.execute("file_read", {
            "file_path": __file__,
            "limit": 5
        })

        if result.get("success"):
            print("✓ Tool executor working")
            print(f"✓ Read {result.get('lines_read', 0)} lines")
        else:
            print(f"⚠️  Tool execution returned: {result}")

        return True

    except Exception as e:
        print(f"❌ Tool executor failed: {e}")
        return False


def test_agentic_chat():
    """Test agentic chat with tool execution."""
    print("\n" + "=" * 80)
    print("STEP 5: Testing Agentic Chat (Full Integration)")
    print("=" * 80)

    try:
        from kautilya.llm_client import KautilyaLLMClient
        from kautilya.tool_executor import ToolExecutor

        client = KautilyaLLMClient()
        executor = ToolExecutor()

        print("Running test query: 'List available LLM providers'\n")

        response_text = ""
        for chunk in client.chat(
            "List available LLM providers",
            tool_executor=executor,
            stream=True,
            max_tool_iterations=3,
        ):
            print(chunk, end="", flush=True)
            response_text += chunk

        print("\n")

        if response_text:
            print("✓ Agentic chat working!")
            print(f"✓ Response length: {len(response_text)} chars")
            return True
        else:
            print("⚠️  No response received")
            return False

    except Exception as e:
        print(f"❌ Agentic chat failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("KAUTILYA LLM CONNECTION DIAGNOSTIC")
    print("=" * 80 + "\n")

    results = []

    # Step 1: Check API key
    if not check_api_key():
        print("\n" + "=" * 80)
        print("DIAGNOSIS: API key not set or invalid")
        print("=" * 80)
        print("\nFix the API key issue above and run this script again.")
        return 1

    results.append(("API Key", True))

    # Step 2: Test OpenAI connection
    if not test_openai_connection():
        print("\n" + "=" * 80)
        print("DIAGNOSIS: Cannot connect to OpenAI API")
        print("=" * 80)
        print("\nFix the connection issue above and try again.")
        return 1

    results.append(("OpenAI Connection", True))

    # Step 3: Test LLM client
    if not test_llm_client():
        print("\n" + "=" * 80)
        print("DIAGNOSIS: LLM client initialization failed")
        print("=" * 80)
        return 1

    results.append(("LLM Client", True))

    # Step 4: Test tool executor
    if not test_tool_executor():
        print("\n" + "=" * 80)
        print("DIAGNOSIS: Tool executor failed")
        print("=" * 80)
        return 1

    results.append(("Tool Executor", True))

    # Step 5: Test full agentic chat
    if not test_agentic_chat():
        print("\n" + "=" * 80)
        print("DIAGNOSIS: Agentic chat integration failed")
        print("=" * 80)
        return 1

    results.append(("Agentic Chat", True))

    # All tests passed
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

    print("\nTest Results:")
    for test_name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {test_name}")

    print("\n" + "=" * 80)
    print("Your Kautilya LLM setup is working correctly!")
    print("=" * 80)
    print("\nYou can now run:")
    print("  kautilya")
    print("\nOr:")
    print("  python -m kautilya.cli")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
