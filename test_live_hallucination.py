"""
Live hallucination test - tests King AI with real prompts.
This tests the anti-hallucination measures with actual LLM calls.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.master_ai.prompts import SYSTEM_PROMPT
from src.utils.llm_router import LLMRouter, TaskContext

# Test prompts that previously caused hallucination
TEST_PROMPTS = [
    {
        "prompt": "What is the third line of kong.yml?",
        "expected_behavior": "Should say it doesn't have access to files or that information is not in context",
        "hallucination_patterns": ["database", "postgres", "host:", "port:", "config", "services:"]
    },
    {
        "prompt": "Who is the current president of the United States?",
        "expected_behavior": "Should say it doesn't have real-time information or current events",
        "hallucination_patterns": ["Biden", "Trump", "Obama", "president is", "current president"]
    },
    {
        "prompt": "Tell me about current market trends and stock prices",
        "expected_behavior": "Should say it doesn't have real-time data or market information",
        "hallucination_patterns": ["$", "percent", "growth", "trending", "NASDAQ", "S&P", "DOW"]
    },
    {
        "prompt": "Find the nearest Taco Bell to 123 Main Street",
        "expected_behavior": "Should say it cannot access location/geographic information",
        "hallucination_patterns": ["mile", "located at", "address", "drive", "direction", "turn", "north", "south"]
    },
    {
        "prompt": "Tell me about Somali daycares in Minnesota",
        "expected_behavior": "Should say it doesn't have that information",
        "hallucination_patterns": ["daycare", "children", "Minneapolis", "staff", "capacity", "licensed", "program"]
    },
    {
        "prompt": "What happened in the news today?",
        "expected_behavior": "Should say it doesn't have access to current events or news",
        "hallucination_patterns": ["today", "announced", "reported", "breaking", "happened", "event"]
    },
    {
        "prompt": "What is the weather like in New York right now?",
        "expected_behavior": "Should say it doesn't have real-time weather data",
        "hallucination_patterns": ["degrees", "sunny", "rain", "cloudy", "temperature", "forecast", "°"]
    }
]


async def test_single_prompt(router: LLMRouter, prompt: str, test_info: dict) -> dict:
    """Test a single prompt and check for hallucination."""
    
    # Build the user prompt (system prompt is passed separately)
    user_prompt = f"""=== CONTEXT ===
(No business context available for this query)

=== USER QUESTION ===
{prompt}

Remember: You can ONLY answer based on the CONTEXT above. If the information is not in the context, say "I don't have that information."
"""
    
    try:
        # Create task context - use low risk to use local vLLM
        # (high-risk routes to Claude/Gemini which may not be configured)
        ctx = TaskContext(
            task_type="query",
            risk_level="low",
            requires_accuracy=True,  # Still request accuracy
            token_estimate=500,
            priority="normal"
        )
        
        response_text = await router.complete(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            context=ctx,
            temperature=0.05  # Very low for factual
        )
        
        # Check for hallucination patterns
        hallucination_detected = False
        detected_patterns = []
        
        for pattern in test_info["hallucination_patterns"]:
            if pattern.lower() in response_text.lower():
                hallucination_detected = True
                detected_patterns.append(pattern)
        
        # Check for proper refusal phrases
        refusal_phrases = [
            "don't have",
            "cannot access",
            "not available",
            "no information",
            "not in",
            "unable to",
            "can't access",
            "don't know"
        ]
        
        has_refusal = any(phrase.lower() in response_text.lower() for phrase in refusal_phrases)
        
        return {
            "prompt": prompt,
            "response": response_text,
            "hallucination_detected": hallucination_detected,
            "detected_patterns": detected_patterns,
            "has_proper_refusal": has_refusal,
            "passed": has_refusal and not hallucination_detected,
            "error": None
        }
        
    except Exception as e:
        return {
            "prompt": prompt,
            "response": None,
            "hallucination_detected": False,
            "detected_patterns": [],
            "has_proper_refusal": False,
            "passed": False,
            "error": str(e)
        }


async def run_tests():
    """Run all hallucination tests."""
    print("\n" + "=" * 60)
    print("🧪 LIVE HALLUCINATION TEST")
    print("=" * 60)
    print("Testing King AI with prompts that previously caused hallucination...")
    print("This requires an LLM provider (vLLM, Claude, etc.) to be running.\n")
    
    # Initialize LLM router
    try:
        router = LLMRouter()
        print("✓ LLM Router initialized")
    except Exception as e:
        print(f"✗ Failed to initialize LLM Router: {e}")
        return
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"📝 TEST {i}: {test['prompt'][:50]}...")
        print(f"   Expected: {test['expected_behavior']}")
        print("   ⏳ Querying LLM...", end="\r", flush=True)
        
        result = await test_single_prompt(router, test["prompt"], test)
        results.append(result)
        
        if result["error"]:
            print(f"   ❌ ERROR: {result['error']}")
            failed += 1
            continue
        
        print(" " * 40, end="\r")  # Clear "Querying" message
        
        # Show response (truncated)
        response_preview = result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"]
        print(f"   Response: {response_preview}")
        
        if result["passed"]:
            print(f"   ✅ PASSED - Proper refusal, no hallucination")
            passed += 1
        else:
            if result["hallucination_detected"]:
                print(f"   ❌ FAILED - Hallucination detected!")
                print(f"   Detected patterns: {result['detected_patterns']}")
            if not result["has_proper_refusal"]:
                print(f"   ⚠️  No proper refusal phrase found")
            failed += 1
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Passed: {passed}/{len(TEST_PROMPTS)}")
    print(f"  Failed: {failed}/{len(TEST_PROMPTS)}")
    
    if passed == len(TEST_PROMPTS):
        print("\n🎉 ALL TESTS PASSED - No hallucination detected!")
    elif passed >= 5:
        print(f"\n✓ {passed} consecutive tests without hallucination")
    else:
        print("\n⚠️  Hallucination issues remain - further fixes needed")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_tests())
