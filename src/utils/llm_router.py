"""
LLM Router - Intelligent routing between inference providers.
Implements hybrid routing with fallback for reliability.
"""

import asyncio
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass
import time
import os

from src.utils.ollama_client import OllamaClient
from src.utils.gemini_client import GeminiClient
from config.settings import settings


class ProviderType(Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"       # Development / primary
    GEMINI = "gemini"       # Cloud fallback


@dataclass
class TaskContext:
    """Context for routing decisions."""
    task_type: str          # "research", "finance", "legal", etc.
    risk_level: str         # "low", "medium", "high"
    requires_accuracy: bool # High-stakes decision?
    token_estimate: int     # Estimated tokens needed
    priority: str           # "normal", "high", "critical"


class LLMRouter:
    """
    Routes inference requests to the optimal provider based on:
    - Task risk level (high-stakes -> more accurate provider)
    - Provider health status
    - Current load and latency
    - Cost optimization
    """
    
    def __init__(self):
        """Initialize all available providers."""
        # Primary: Ollama for production
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        # Secondary: Gemini for cloud fallback
        self.gemini: Optional[GeminiClient] = None
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.gemini = GeminiClient(api_key=gemini_key)
        
        # Health tracking
        self._provider_health = {
            ProviderType.OLLAMA: True,
            ProviderType.GEMINI: True if self.gemini else False,
        }
        
        # Circuit breaker state
        self._failure_counts = {p: 0 for p in ProviderType}
        self._circuit_open = {p: False for p in ProviderType}
        self._circuit_open_until = {p: 0 for p in ProviderType}
        self._failure_threshold = 3
        self._circuit_timeout = 60  # seconds
    
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        context: TaskContext | None = None
    ) -> str:
        """
        Route and execute an inference request.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            context: Task context for routing decisions
            
        Returns:
            Generated response from selected provider
        """
        # Determine routing - use Gemini for high-stakes, Ollama otherwise
        if context and context.risk_level == "high" and context.requires_accuracy:
            if self.gemini and self._provider_health[ProviderType.GEMINI]:
                providers = [ProviderType.GEMINI, ProviderType.OLLAMA]
            else:
                providers = [ProviderType.OLLAMA]
        else:
            providers = [ProviderType.OLLAMA, ProviderType.GEMINI] if self.gemini else [ProviderType.OLLAMA]
        
        # Execute with fallback chain
        last_error = None
        for provider in providers:
            if self._is_circuit_open(provider):
                continue
                
            try:
                result = await self._execute(provider, prompt, system)
                
                # Record success
                self._record_success(provider)
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(provider)
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def _execute(
        self,
        provider: ProviderType,
        prompt: str,
        system: str | None
    ) -> str:
        """Execute inference on specific provider."""
        if provider == ProviderType.OLLAMA:
            return await self.ollama.complete(prompt, system)
        elif provider == ProviderType.GEMINI and self.gemini:
            return await self.gemini.complete(prompt, system)
        else:
            raise ValueError(f"Provider {provider} not available")
    
    def _is_circuit_open(self, provider: ProviderType) -> bool:
        """Check if circuit breaker is open for provider."""
        if not self._circuit_open[provider]:
            return False
        
        # Check if timeout has passed
        if time.time() > self._circuit_open_until[provider]:
            self._circuit_open[provider] = False
            self._failure_counts[provider] = 0
            return False
        
        return True
    
    def _record_success(self, provider: ProviderType):
        """Record successful request."""
        self._failure_counts[provider] = 0
        self._circuit_open[provider] = False
        self._provider_health[provider] = True
    
    def _record_failure(self, provider: ProviderType):
        """Record failed request and potentially open circuit."""
        self._failure_counts[provider] += 1
        
        if self._failure_counts[provider] >= self._failure_threshold:
            self._circuit_open[provider] = True
            self._circuit_open_until[provider] = time.time() + self._circuit_timeout
            self._provider_health[provider] = False
    
    async def health_check_all(self) -> dict:
        """Check health of all providers."""
        results = {}
        
        try:
            await self.ollama.complete("test", None)
            results[ProviderType.OLLAMA] = True
        except:
            results[ProviderType.OLLAMA] = False
        
        if self.gemini:
            try:
                await self.gemini.complete("test", None)
                results[ProviderType.GEMINI] = True
            except:
                results[ProviderType.GEMINI] = False
        else:
            results[ProviderType.GEMINI] = False
        
        self._provider_health = results
        return results
