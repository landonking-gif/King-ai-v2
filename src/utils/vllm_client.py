"""
vLLM Client for high-throughput production inference.
Provides batched inference with OpenAI-compatible API.
"""

import httpx
import asyncio
from typing import AsyncIterator, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7
    request_id: str = None


class VLLMClient:
    """
    Async client for vLLM's OpenAI-compatible API.
    Supports batched requests for high throughput.
    """
    
    def __init__(
        self,
        base_url: str,
        model: str = "meta-llama/Llama-3.1-70B-Instruct",
        max_concurrent: int = 10
    ):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: vLLM server URL (e.g., http://inference-alb.internal:8080)
            model: Model identifier for the API
            max_concurrent: Maximum concurrent requests
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = httpx.AsyncClient(timeout=300.0)
        
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a completion using vLLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate (dynamically capped based on input)
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Estimate input tokens (~4 chars per token) and cap max_tokens dynamically
        # Model context is 4096, leave buffer for safety
        total_input_chars = len(prompt) + (len(system) if system else 0)
        estimated_input_tokens = total_input_chars // 3  # Conservative estimate
        available_tokens = 4096 - estimated_input_tokens - 100  # 100 token safety buffer
        capped_max_tokens = max(256, min(max_tokens, available_tokens))  # At least 256 tokens
        
        async with self.semaphore:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": capped_max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def batch_complete(
        self,
        requests: List[InferenceRequest]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inference requests concurrently.
        
        Args:
            requests: List of InferenceRequest objects
            
        Returns:
            List of results with request_id and response
        """
        async def process_single(req: InferenceRequest) -> Dict[str, Any]:
            try:
                response = await self.complete(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature
                )
                return {
                    "request_id": req.request_id,
                    "success": True,
                    "response": response
                }
            except Exception as e:
                return {
                    "request_id": req.request_id,
                    "success": False,
                    "error": str(e)
                }
        
        tasks = [process_single(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def complete_stream(
        self,
        prompt: str,
        system: str | None = None
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            
        Yields:
            Generated tokens as they arrive
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Estimate input tokens and cap max_tokens dynamically
        total_input_chars = len(prompt) + (len(system) if system else 0)
        estimated_input_tokens = total_input_chars // 3
        available_tokens = 4096 - estimated_input_tokens - 100
        capped_max_tokens = max(256, min(1024, available_tokens))
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": capped_max_tokens,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def aclose(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
