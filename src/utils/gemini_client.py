
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

    async def complete(self, prompt: str, system: str | None = None) -> str:
        """Generates a text completion using Gemini."""
        payload = {
            "contents": [{
                "parts": [{"text": f"{system}\n\nUser: {prompt}" if system else prompt}]
            }],
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            response = await self.client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Error connecting to Gemini: {str(e)}"

    async def health_check(self) -> bool:
        """Checks if the Gemini API key is valid by sending a tiny prompt."""
        try:
            res = await self.complete("Hi", "Return 'ok'")
            return "ok" in res.lower()
        except:
            return False
