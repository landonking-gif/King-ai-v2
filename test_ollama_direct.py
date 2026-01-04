"""
Direct Ollama test - bypasses router to test anti-hallucination prompts.
"""
import asyncio
import httpx

async def test_ollama_connection():
    """Test that Ollama is reachable."""
    print("=" * 60)
    print("TESTING OLLAMA CONNECTION (using llama3.2:1b for speed)")
    print("=" * 60)
    
    client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    
    try:
        response = await client.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:1b',  # Use smaller faster model
                'prompt': 'Say exactly these words: I am working correctly.',
                'stream': False
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', 'No response')[:300]}")
            print("\n[OK] Ollama is working!")
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
    result = asyncio.run(test_ollama_connection())
    if not result:
        print("\n[FAIL] Ollama is NOT working. Please start Ollama first.")
    else:
        print("\nOllama is ready for testing.")
