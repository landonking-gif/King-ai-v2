
import httpx
import json
import os
from typing import Optional

class GeminiClient:
    """
    Async client for Google's Gemini API.
    Used as a fallback when local/EC2 Ollama is unavailable.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        self.client = httpx.AsyncClient(timeout=60.0)

    async def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """Generates a text completion using Gemini."""
        payload = {
            "contents": [{
                "parts": [{"text": f"{system}\n\nUser: {prompt}" if system else prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            response = await self.client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' not in data or not data['candidates']:
                if 'error' in data:
                    error_msg = data['error'].get('message', 'Unknown Gemini error')
                    raise RuntimeError(f"Gemini API error: {error_msg}")
                raise RuntimeError(f"Invalid Gemini response format: {json.dumps(data)}")
                
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            # Re-raise so LLMRouter can handle fallback
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                except:
                    error_msg = str(e)
                raise RuntimeError(f"Gemini API error ({e.response.status_code}): {error_msg}")
            raise

    async def health_check(self) -> bool:
        """Checks if the Gemini API key is valid by sending a tiny prompt."""
        try:
            res = await self.complete("Hi", "Return 'ok'")
            return "ok" in res.lower()
        except:
            return False

    async def aclose(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()
