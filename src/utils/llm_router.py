"""
LLM Router - Intelligent routing between inference providers.
Implements hybrid routing with fallback for reliability.
"""

import asyncio
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass
import time
import os

from src.utils.ollama_client import OllamaClient
from src.utils.vllm_client import VLLMClient
from src.utils.gemini_client import GeminiClient
from src.utils.claude_client import ClaudeClient
from config.settings import settings


class ProviderType(Enum):
    """Available LLM providers."""
    VLLM = "vllm"           # High-throughput production
    OLLAMA = "ollama"       # Development / fallback
    GEMINI = "gemini"       # Cloud fallback
    CLAUDE = "claude"       # High-stakes fallback (future)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    provider: ProviderType
    reason: str
    latency_ms: float = 0


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
        # Primary: vLLM for production throughput
        self.vllm: Optional[VLLMClient] = None
        if hasattr(settings, 'vllm_url') and settings.vllm_url:
            self.vllm = VLLMClient(
                base_url=settings.vllm_url,
                model=settings.vllm_model if hasattr(settings, 'vllm_model') else "meta-llama/Llama-3.1-70B-Instruct"
            )
        
        # Secondary: Ollama for dev/fallback
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        # Tertiary: Gemini for cloud fallback
        self.gemini: Optional[GeminiClient] = None
        if settings.gemini_api_key:
            self.gemini = GeminiClient(api_key=settings.gemini_api_key)

        # High-stakes fallback: Claude
        self.claude: Optional[ClaudeClient] = None
        if settings.anthropic_api_key:
            self.claude = ClaudeClient(
                api_key=settings.anthropic_api_key,
                model=settings.claude_model or "claude-3-5-sonnet-20241022",
            )
        
        # Health tracking
        self._provider_health = {p: True for p in ProviderType}
        self._provider_health[ProviderType.VLLM] = bool(self.vllm)
        self._provider_health[ProviderType.GEMINI] = bool(self.gemini)
        self._provider_health[ProviderType.CLAUDE] = bool(self.claude and self.claude.is_available())
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
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
        # Determine routing
        decision = await self._route(context)
        
        # Execute with fallback chain
        providers = self._get_fallback_chain(decision.provider)
        
        last_error = None
        for provider in providers:
            if not self._provider_health.get(provider, False):
                continue
            if self._is_circuit_open(provider):
                continue
                
            try:
                start = time.time()
                result = await self._execute(provider, prompt, system)
                latency = (time.time() - start) * 1000
                
                # Record success
                self._record_success(provider)
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(provider)
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def _route(self, context: TaskContext | None) -> RoutingDecision:
        """Determine the best provider for this request."""

        # High-stakes: prefer Claude (then Gemini) when configured
        if context and context.risk_level == "high" and context.requires_accuracy:
            if self.claude and self._provider_health[ProviderType.CLAUDE] and self.claude.is_available():
                return RoutingDecision(
                    provider=ProviderType.CLAUDE,
                    reason="High-stakes task routed to Claude for accuracy"
                )
            if self.gemini and self._provider_health[ProviderType.GEMINI]:
                return RoutingDecision(
                    provider=ProviderType.GEMINI,
                    reason="High-stakes task routed to Gemini for accuracy"
                )
        
        # Default to vLLM if available and healthy
        if self.vllm and self._provider_health[ProviderType.VLLM]:
            return RoutingDecision(
                provider=ProviderType.VLLM,
                reason="Primary production provider"
            )
        
        # Fallback to Ollama
        if self._provider_health[ProviderType.OLLAMA]:
            return RoutingDecision(
                provider=ProviderType.OLLAMA,
                reason="Fallback to Ollama (vLLM unavailable)"
            )
        
        # Last resort: Gemini
        if self.gemini and self._provider_health[ProviderType.GEMINI]:
            return RoutingDecision(
                provider=ProviderType.GEMINI,
                reason="Cloud fallback (all local providers down)"
            )

        # Last resort: Claude (only if configured)
        if self.claude and self._provider_health[ProviderType.CLAUDE] and self.claude.is_available():
            return RoutingDecision(
                provider=ProviderType.CLAUDE,
                reason="Cloud fallback (all other providers down)"
            )
        
        raise RuntimeError("No healthy providers available")
    
    def _get_fallback_chain(self, primary: ProviderType) -> list[ProviderType]:
        """Get ordered fallback chain starting from primary."""
        if primary == ProviderType.CLAUDE:
            # High-stakes path: allow other providers as fallback
            return [
                ProviderType.CLAUDE,
                ProviderType.GEMINI,
                ProviderType.VLLM,
                ProviderType.OLLAMA,
            ]

        if primary == ProviderType.GEMINI:
            return [ProviderType.GEMINI, ProviderType.VLLM, ProviderType.OLLAMA]

        if primary == ProviderType.OLLAMA:
            return [ProviderType.OLLAMA, ProviderType.VLLM, ProviderType.GEMINI]

        # Default: VLLM -> Ollama -> Gemini (keep Claude out of normal fallback chain)
        return [ProviderType.VLLM, ProviderType.OLLAMA, ProviderType.GEMINI]
    
    async def _execute(
        self,
        provider: ProviderType,
        prompt: str,
        system: str | None
    ) -> str:
        """Execute inference on specific provider."""
        if provider == ProviderType.VLLM and self.vllm:
            return await self.vllm.complete(prompt, system)
        elif provider == ProviderType.OLLAMA:
            return await self.ollama.complete(prompt, system)
        elif provider == ProviderType.GEMINI and self.gemini:
            return await self.gemini.complete(prompt, system)
        elif provider == ProviderType.CLAUDE and self.claude:
            return await self.claude.complete(prompt, system=system)
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
    
    async def health_check_all(self) -> dict[ProviderType, bool]:
        """Check health of all providers."""
        results = {}

        # Avoid making external network calls during test runs.
        # (Some developers may have API keys set in their environment.)
        in_tests = bool(os.getenv("PYTEST_CURRENT_TEST"))
        
        if self.vllm:
            results[ProviderType.VLLM] = await self.vllm.health_check()
        else:
            results[ProviderType.VLLM] = False
            
        results[ProviderType.OLLAMA] = await self.ollama.health_check()
        
        if self.gemini:
            if in_tests:
                results[ProviderType.GEMINI] = True
            else:
                try:
                    # Simple test
                    await self.gemini.complete("test", None)
                    results[ProviderType.GEMINI] = True
                except:
                    results[ProviderType.GEMINI] = False
        else:
            results[ProviderType.GEMINI] = False

        if self.claude and self.claude.is_available():
            if in_tests:
                results[ProviderType.CLAUDE] = True
            else:
                try:
                    await self.claude.complete("test", system=None, max_tokens=16, temperature=0)
                    results[ProviderType.CLAUDE] = True
                except:
                    results[ProviderType.CLAUDE] = False
        else:
            results[ProviderType.CLAUDE] = False
        
        self._provider_health = results
        return results
