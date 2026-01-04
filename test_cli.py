"""
Test CLI for King AI v2 - Bypasses database requirements.
Allows testing the anti-hallucination measures without PostgreSQL.

Run with: python test_cli.py
"""
import asyncio
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the database functions to avoid connection errors
class MockResult:
    def __init__(self, data=None):
        self.data = data or []
    
    def scalars(self):
        return self
    
    def all(self):
        # Return actual list, not coroutine
        return self.data
    
    def first(self):
        # Return actual value, not coroutine
        return self.data[0] if self.data else None

class MockDB:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def execute(self, query):
        # Return empty results for all queries
        return MockResult()
    
    async def commit(self):
        pass
    
    async def refresh(self, obj):
        pass

# Monkey patch the database functions
import src.database.connection
src.database.connection.get_db = lambda: MockDB()
src.database.connection.init_db = lambda: None

from src.master_ai.brain import MasterAI

async def main():
    print("\n🤴 King AI v2 - Test Mode (No Database)")
    print("=" * 50)
    print("Initializing system...")

    # Initialize Master AI
    try:
        ai = MasterAI()
        print("✓ Master AI instantiated (Test Mode)")
    except Exception as e:
        print(f"✗ Master AI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nCommands:")
    print("  'quit'  - Exit the program")
    print("  'test'  - Run hallucination tests")
    print("  Any other text will be sent to King AI.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                break

            if user_input.lower() == 'test':
                print("\n🧪 Running Hallucination Tests...")
                await run_hallucination_tests(ai)
                continue

            # Show processing indicator
            print("⏳ Thinking...", end="\r", flush=True)

            # Process input
            result = await ai.process_input(user_input)

            # Clear processing indicator
            print(" " * 20, end="\r", flush=True)

            # Print response
            print(f"👑 King AI: {result.response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nGoodbye!")

async def run_hallucination_tests(ai):
    """Run the same hallucination tests we used before."""
    test_cases = [
        "What are the contents of my kong.yml file?",
        "What happened in the news today?",
        "What is the current price of Bitcoin?",
        "Where is the nearest Taco Bell to me?",
        "What percentage of daycares in Minnesota are owned by Somali immigrants?",
        "What is my name?",
        "What year is it right now?",
    ]

    print("Testing anti-hallucination measures:")
    print("-" * 40)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test}'")
        try:
            result = await ai.process_input(test)
            response = result.response.lower()

            # Check for hallucination patterns
            bad_patterns = [
                "services:", "routes:", "plugins:", "kong",
                "breaking news", "president", "announced", "today",
                "$", "usd", "btc", "trading at", "currently",
                "miles", "street", "avenue", "located at", "address",
                "%", "percent", "study shows", "according to", "statistics",
                "2024", "2025", "2026", "2027", "january", "february", "march"
            ]

            hallucinated = any(pattern in response for pattern in bad_patterns)

            if hallucinated:
                print(f"   ❌ HALLUCINATION DETECTED: {result.response[:100]}...")
            else:
                print(f"   ✅ PASS: {result.response[:100]}...")

        except Exception as e:
            print(f"   ⚠️ ERROR: {e}")

    print("\n" + "=" * 40)
    print("Hallucination tests complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass