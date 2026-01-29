"""
Direct vLLM test - bypasses router to test anti-hallucination prompts.
"""
import asyncio
import httpx

async def test_vllm_connection():
    """Test that vLLM is reachable."""
    print("=" * 60)
    print("TESTING VLLM CONNECTION (using Kimi-K2-Thinking)")
    print("=" * 60)

    client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout

    try:
        response = await client.post(
            'http://localhost:8005/v1/chat/completions',
            json={
                'model': 'moonshotai/Kimi-K2-Thinking',
                'messages': [{'role': 'user', 'content': 'Say exactly these words: I am working correctly.'}],
                'max_tokens': 50
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
            print(f"Response: {content[:300]}")
            print("\n[OK] vLLM is working!")
            return True
        else:
            print(f"Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        return False
    finally:
        await client.aclose()

if __name__ == "__main__":
    result = asyncio.run(test_vllm_connection())
    if not result:
        print("\n[FAIL] vLLM is NOT working. Please start vLLM first.")
    else:
        print("\nvLLM is ready for testing.")
