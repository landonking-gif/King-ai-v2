
import asyncio
import sys
import os
import json
from unittest.mock import MagicMock, AsyncMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.llm_router import LLMRouter, ProviderType, TaskContext
from src.utils.gemini_client import GeminiClient
from src.utils.ollama_client import OllamaClient
from config.settings import settings

async def test_gemini_fallback():
    print("Testing Gemini fallback...")
    
    # 1. Setup Router
    # Mock settings to have Gemini as primary
    with patch('src.utils.llm_router.settings') as mock_settings:
        mock_settings.primary_model = "gemini"
        mock_settings.gemini_api_key = "valid_key"
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.1:8b"
        mock_settings.vllm_url = None
        
        router = LLMRouter()
        
        # 2. Mock Gemini to fail with 400
        mock_gemini = AsyncMock(spec=GeminiClient)
        mock_gemini.complete.side_effect = RuntimeError("Gemini API error (400): Bad Request")
        router.gemini = mock_gemini
        router._provider_health[ProviderType.GEMINI] = True
        
        # 3. Mock Ollama to succeed
        mock_ollama = AsyncMock(spec=OllamaClient)
        mock_ollama.complete.return_value = "This is a fallback response from Ollama"
        router.ollama = mock_ollama
        router._provider_health[ProviderType.OLLAMA] = True
        
        # 4. Execute completion
        try:
            result = await router.complete("Hello", context=TaskContext(
                task_type="conversation",
                risk_level="low",
                requires_accuracy=False,
                token_estimate=100,
                priority="normal"
            ))
            print(f"Result: {result}")
            if result == "This is a fallback response from Ollama":
                print("SUCCESS: Fallback worked!")
            else:
                print("FAILURE: Unexpected result.")
        except Exception as e:
            print(f"FAILURE: Exception leaked: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_fallback())
