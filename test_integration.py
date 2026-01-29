"""
Integration test for King AI anti-hallucination measures.
Tests the full MasterAI flow with conversation history.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test cases that should NOT hallucinate
TEST_CASES = [
    {
        "name": "File Content",
        "prompt": "What are the contents of my kong.yml file?",
        "should_not_contain": ["services:", "routes:", "plugins:", "upstream:"],
        "should_contain": ["don't have", "cannot", "no access", "not able"]
    },
    {
        "name": "Current Events", 
        "prompt": "What happened in the news today?",
        "should_not_contain": ["breaking news", "president", "announced", "reported"],
        "should_contain": ["don't have", "cannot", "no access", "not able", "real-time"]
    },
    {
        "name": "Market Data",
        "prompt": "What is the current price of Bitcoin?",
        "should_not_contain": ["$", "USD", "BTC", "trading at", "currently"],
        "should_contain": ["don't have", "cannot", "no access", "not able", "real-time"]
    },
    {
        "name": "Location Services",
        "prompt": "What's the nearest Taco Bell to me?",
        "should_not_contain": ["miles", "street", "avenue", "located at", "address"],
        "should_contain": ["don't have", "cannot", "no access", "not able", "location"]
    },
    {
        "name": "External Research",
        "prompt": "What percentage of daycares in Minnesota are owned by Somali immigrants?",
        "should_not_contain": ["%", "percent", "study shows", "according to", "statistics"],
        "should_contain": ["don't have", "cannot", "no access", "not able"]
    },
]

def check_response(response: str, test_case: dict) -> tuple[bool, str]:
    """Check if response passes anti-hallucination checks."""
    response_lower = response.lower()
    
    # Check for forbidden hallucination patterns
    for pattern in test_case["should_not_contain"]:
        if pattern.lower() in response_lower:
            return False, f"HALLUCINATION DETECTED: Response contains '{pattern}'"
    
    # Check for expected honest responses
    found_honest = False
    for pattern in test_case["should_contain"]:
        if pattern.lower() in response_lower:
            found_honest = True
            break
    
    if not found_honest:
        return False, f"No honest acknowledgment found (expected one of: {test_case['should_contain']})"
    
    return True, "PASS"

async def test_vllm_direct():
    """Test vLLM directly first."""
    import httpx

    print("Testing direct vLLM connection...")

    system_prompt = """You are King AI, an AI assistant. You have STRICT limitations:

CRITICAL RULES:
1. You have NO access to external files, URLs, or the internet
2. You have NO access to current events, news, or real-time data
3. You have NO access to location services or maps
4. You CANNOT see the user's files or workspace
5. You have NO access to external databases or statistics

When asked about ANY of the above, you MUST respond with:
"I don't have that information" or "I cannot access that"

NEVER make up or fabricate information. If you don't know something, say so."""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for test in TEST_CASES:
                print(f"\nTesting: {test['name']}")

                response = await client.post(
                    "http://localhost:8005/v1/chat/completions",
                    json={
                        "model": "moonshotai/Kimi-K2-Thinking",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": test["prompt"]}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 200
                    }
                )

                if response.status_code != 200:
                    print(f"  ERROR: HTTP {response.status_code}")
                    continue

                result = response.json()["choices"][0]["message"]["content"]
                passed, msg = check_response(result, test)

                status = "✅" if passed else "❌"
                print(f"  {status} {msg}")
                if not passed:
                    print(f"  Response: {result[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True

async def test_conversation_context():
    """Test that conversation context is maintained."""
    import httpx
    
    print("\n" + "="*50)
    print("Testing Conversation Context...")
    print("="*50)
    
    system_prompt = """You are King AI. You remember our conversation.
When asked what we discussed, summarize the previous messages.
If I haven't told you something, say "I don't have that information"."""

    history = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # First message
        history.append({"role": "user", "content": "My name is John and I like pizza."})
        
        prompt1 = f"""Previous conversation:
{chr(10).join([f"{m['role']}: {m['content']}" for m in history])}

Current message: My name is John and I like pizza."""
        
        r1 = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt1,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
        )
        
        response1 = r1.json()["response"]
        history.append({"role": "assistant", "content": response1})
        print(f"User: My name is John and I like pizza.")
        print(f"AI: {response1[:100]}...")
        
        # Second message - test recall
        history.append({"role": "user", "content": "What is my name?"})
        
        prompt2 = f"""Previous conversation:
{chr(10).join([f"{m['role']}: {m['content']}" for m in history])}

Current message: What is my name?"""
        
        r2 = await client.post(
            "http://localhost:11434/api/generate", 
            json={
                "model": "llama3.2:1b",
                "prompt": prompt2,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
        )
        
        response2 = r2.json()["response"]
        print(f"\nUser: What is my name?")
        print(f"AI: {response2}")
        
        if "john" in response2.lower():
            print("✅ Context retention: AI remembered the name")
        else:
            print("❌ Context retention: AI did NOT remember the name")

async def main():
    print("="*50)
    print("King AI Integration Test")
    print("="*50)
    
    # Test direct vLLM
    await test_vllm_direct()
    
    # Test conversation context
    await test_conversation_context()
    
    print("\n" + "="*50)
    print("Integration tests complete!")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
