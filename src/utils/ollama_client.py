"""
Ollama API client for LLM inference.
This module provides an asynchronous wrapper around the Ollama REST API
to facilitate generation, streaming, and model management.
"""

import httpx
from typing import AsyncIterator
import json

class OllamaClient:
    """
    Async client for interfacting with an Ollama server.
    Ensures long-running generation tasks don't block the system.
    """
    
    def __init__(self, base_url: str, model: str):
        """
        Initialize the client with connection details.
        :param base_url: The URL where Ollama is hosted.
        :param model: The default LLM model to use for generation.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 5 minute timeout to accommodate large model cold starts or long outputs
        self.client = httpx.AsyncClient(timeout=300.0) 
    
    async def complete(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        """
        Generate a full completion for the given prompt.
        
        Args:
            prompt: The main user-provided instructions or query.
            system: Optional system prompt to override the default model behavior.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
            
        Returns:
            The raw text response from the model.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False, # Wait for the full response before returning
            "options": {
                "temperature": temperature
            }
        }
        
        if system:
            payload["system"] = system
        
        # Post request to the generation endpoint
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status() # Raise error for 4xx/5xx responses
        
        return response.json()["response"]
    
    async def complete_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream an LLM completion token by token using an asynchronous iterator.
        Useful for building real-time chat interfaces.
        
        Args:
            prompt: The user input to generate from.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
    
    async def list_models(self) -> list[str]:
        """
        Queries the Ollama server for a list of locally available models.
        :return: A list of model names (strings).
        """
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    
    async def health_check(self) -> bool:
        """
        Checks if the Ollama server is up and responding.
        :return: True if healthy, False otherwise.
        """
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    async def aclose(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()
