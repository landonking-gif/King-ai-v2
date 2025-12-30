"""
Simple LLM Router for routing inference requests.
Initially supports Ollama, with extensibility for other providers.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.utils.ollama_client import OllamaClient
from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("llm_router")


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
    Routes inference requests to the optimal LLM provider.
    Currently supports Ollama with extensibility for future providers.
    """
    
    def __init__(self):
        """Initialize the router with available providers."""
        # Primary: Ollama for development and production
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        logger.info(
            "LLM Router initialized",
            ollama_url=settings.ollama_url,
            model=settings.ollama_model
        )
    
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
        # For now, route everything to Ollama
        # Future: Add logic to select between providers based on context
        
        try:
            result = await self.ollama.complete(prompt, system)
            return result
        except Exception as e:
            logger.error("LLM inference failed", error=str(e), exc_info=True)
            raise
    
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy."""
        try:
            return await self.ollama.health_check()
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
