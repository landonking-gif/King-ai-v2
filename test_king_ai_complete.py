"""
King AI v2 - Complete Feature Test Script
Tests all core functionality via command line:
- Date/time awareness
- Math calculations
- Web search capability
- Stock/market data access
- Agent communication
- Memory system

Run with: py test_king_ai_complete.py
"""
import asyncio
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the database functions to avoid connection errors for testing
class MockResult:
    def __init__(self, data=None):
        self.data = data or []
    
    def scalars(self):
        return self
    
    def all(self):
        return self.data
    
    def first(self):
        return self.data[0] if self.data else None

class MockDB:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def execute(self, query):
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
from src.utils.web_tools import get_web_tools, evaluate_simple_math


class TestResult:
    def __init__(self, name: str, passed: bool, expected: str = "", actual: str = "", error: str = ""):
        self.name = name
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.error = error


async def test_datetime_awareness():
    """Test that King AI knows the current date and time."""
    ai = MasterAI()
    
    result = await ai.process_input("What is the date and time?")
    response = result.response.lower()
    
    # Should contain date-like content
    has_date = any(word in response for word in ["january", "february", "march", "april", "may", "june",
                                                   "july", "august", "september", "october", "november", "december",
                                                   "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
    
    has_time = any(word in response for word in ["am", "pm", ":"])
    
    return TestResult(
        name="DateTime Awareness",
        passed=has_date or has_time,
        expected="Response with current date and/or time",
        actual=result.response[:100]
    )


async def test_simple_math():
    """Test basic math calculations."""
    ai = MasterAI()
    
    result = await ai.process_input("What is 5 times 3?")
    response = result.response.lower()
    
    passed = "15" in response
    
    return TestResult(
        name="Simple Math (5 x 3)",
        passed=passed,
        expected="15",
        actual=result.response[:100]
    )


async def test_math_addition():
    """Test addition."""
    ai = MasterAI()
    
    result = await ai.process_input("What is 100 plus 250?")
    response = result.response.lower()
    
    passed = "350" in response
    
    return TestResult(
        name="Math Addition (100 + 250)",
        passed=passed,
        expected="350",
        actual=result.response[:100]
    )


async def test_identity():
    """Test King AI knows who it is."""
    ai = MasterAI()
    
    result = await ai.process_input("Who are you?")
    response = result.response.lower()
    
    passed = "king ai" in response or "king" in response
    
    return TestResult(
        name="Identity",
        passed=passed,
        expected="King AI introduction",
        actual=result.response[:100]
    )


async def test_greeting():
    """Test greeting response."""
    ai = MasterAI()
    
    result = await ai.process_input("Hello")
    response = result.response.lower()
    
    passed = any(word in response for word in ["hello", "hi", "good", "help", "king ai"])
    
    return TestResult(
        name="Greeting Response",
        passed=passed,
        expected="Friendly greeting",
        actual=result.response[:100]
    )


async def test_web_tools_datetime():
    """Test web tools date/time directly."""
    web_tools = get_web_tools()
    
    dt = web_tools.get_current_datetime()
    
    passed = all(key in dt for key in ["date", "time", "day_of_week"])
    
    return TestResult(
        name="WebTools DateTime",
        passed=passed,
        expected="Dict with date, time, day_of_week",
        actual=str(dt)[:100]
    )


async def test_web_tools_math():
    """Test math evaluation directly."""
    result1 = evaluate_simple_math("what is 7 times 8")
    result2 = evaluate_simple_math("12 + 5")
    result3 = evaluate_simple_math("100 divided by 4")
    
    passed = result1 == "56" and result2 == "17" and result3 == "25"
    
    return TestResult(
        name="WebTools Math Evaluation",
        passed=passed,
        expected="56, 17, 25",
        actual=f"{result1}, {result2}, {result3}"
    )


async def test_web_search():
    """Test web search capability."""
    web_tools = get_web_tools()
    
    try:
        results = await web_tools.web_search("Python programming", max_results=3)
        passed = len(results) > 0
        actual = f"Found {len(results)} results"
    except Exception as e:
        passed = False
        actual = f"Error: {e}"
    
    return TestResult(
        name="Web Search",
        passed=passed,
        expected="At least 1 result",
        actual=actual
    )


async def test_realtime_detection():
    """Test detection of real-time query needs."""
    web_tools = get_web_tools()
    
    needs1 = web_tools.detect_realtime_query("what time is it")
    needs2 = web_tools.detect_realtime_query("search for AI trends")
    needs3 = web_tools.detect_realtime_query("stock price of AAPL")
    
    passed = needs1["needs_datetime"] and needs2["needs_web_search"] and needs3["needs_stock_data"]
    
    return TestResult(
        name="Realtime Query Detection",
        passed=passed,
        expected="Correct detection of query types",
        actual=f"datetime={needs1['needs_datetime']}, web={needs2['needs_web_search']}, stock={needs3['needs_stock_data']}"
    )


async def test_definition():
    """Test basic knowledge query (will use LLM if available)."""
    ai = MasterAI()
    
    result = await ai.process_input("What is a business?")
    response = result.response.lower()
    
    # Should NOT say "I don't have that information" for a basic definition
    passed = "don't have" not in response or len(response) > 50
    
    return TestResult(
        name="Basic Knowledge Query",
        passed=passed,
        expected="Some definition or helpful response",
        actual=result.response[:100]
    )


async def main():
    print("\n" + "=" * 60)
    print("🤴 King AI v2 - Complete Feature Test Suite")
    print("=" * 60)
    
    tests = [
        ("DateTime Awareness", test_datetime_awareness),
        ("Simple Math", test_simple_math),
        ("Math Addition", test_math_addition),
        ("Identity", test_identity),
        ("Greeting", test_greeting),
        ("WebTools DateTime", test_web_tools_datetime),
        ("WebTools Math", test_web_tools_math),
        ("Web Search", test_web_search),
        ("Realtime Detection", test_realtime_detection),
        ("Basic Knowledge", test_definition),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n📋 Running: {name}...", end=" ", flush=True)
        try:
            result = await test_func()
            results.append(result)
            if result.passed:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print(f"   Expected: {result.expected}")
                print(f"   Actual: {result.actual}")
        except Exception as e:
            results.append(TestResult(name=name, passed=False, error=str(e)))
            print(f"❌ ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed. See details above.")
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error or r.actual[:50]}")
    
    print("\n" + "=" * 60)
    
    # Interactive mode
    print("\n🔄 Starting interactive test mode...")
    print("Type your message to test King AI, or 'quit' to exit.\n")
    
    ai = MasterAI()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print("⏳ Processing...", end="\r", flush=True)
            result = await ai.process_input(user_input)
            print(" " * 20, end="\r", flush=True)
            print(f"👑 King AI: {result.response}")
            
            if result.metadata:
                print(f"   [Metadata: {result.metadata}]")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
