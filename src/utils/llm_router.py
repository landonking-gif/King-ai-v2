"""
LLM Router - Routes LLM requests to appropriate models based on task context.
Provides intelligent model selection and fallback capabilities.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.ollama_client import OllamaClient
from src.utils.structured_logging import get_logger

logger = get_logger("llm_router")


class ModelTier(str, Enum):
    """Model tier for different task complexities."""
    FAST = "fast"          # Quick, simple tasks
    BALANCED = "balanced"  # General purpose
    POWERFUL = "powerful"  # Complex reasoning tasks


@dataclass
class TaskContext:
    """Context information for LLM task routing."""
    task_type: str
    risk_level: str = "low"  # low, medium, high
    requires_accuracy: bool = False
    token_estimate: int = 1000
    priority: str = "normal"  # low, normal, high
    
    def get_recommended_tier(self) -> ModelTier:
        """Determine recommended model tier based on context."""
        if self.risk_level == "high" or self.requires_accuracy:
            return ModelTier.POWERFUL
        elif self.token_estimate > 2000 or self.priority == "high":
            return ModelTier.BALANCED
        else:
            return ModelTier.FAST


class LLMRouter:
    """
    Routes LLM requests to appropriate models.
    Handles model selection, fallback, and error recovery.
    """
    
    def __init__(self, ollama_client: OllamaClient = None, base_url: str = "http://localhost:11434", default_model: str = "llama3.2:latest"):
        """
        Initialize the router.
        
        Args:
            ollama_client: OllamaClient instance, creates default if not provided
            base_url: Ollama server URL (used if creating default client)
            default_model: Default model name (used if creating default client)
        """
        self.ollama = ollama_client or OllamaClient(base_url=base_url, model=default_model)
        
        # Model configuration by tier
        self.model_config = {
            ModelTier.FAST: {
                "model": "llama3.2:latest",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            ModelTier.BALANCED: {
                "model": "llama3.2:latest",
                "temperature": 0.5,
                "max_tokens": 4000
            },
            ModelTier.POWERFUL: {
                "model": "llama3.2:latest",
                "temperature": 0.3,
                "max_tokens": 8000
            }
        }
    
    async def complete(
        self,
        prompt: str,
        context: Optional[TaskContext] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Complete a prompt using appropriate model.
        
        Args:
            prompt: The prompt to complete
            context: Task context for routing decisions
            model: Override model selection
            **kwargs: Additional parameters for the model
            
        Returns:
            Model response as string
        """
        # Determine which model to use
        if model is None:
            if context:
                tier = context.get_recommended_tier()
                config = self.model_config[tier]
                model = config["model"]
                
                # Merge config with kwargs
                for key, value in config.items():
                    if key != "model" and key not in kwargs:
                        kwargs[key] = value
            else:
                # Default to balanced
                model = self.model_config[ModelTier.BALANCED]["model"]
        
        logger.info(f"Routing to model: {model}", task_type=context.task_type if context else "unknown")
        
        try:
            # Note: OllamaClient.complete doesn't accept model parameter
            # It uses the model set during initialization
            # For now, we'll use the client's default model
            response = await self.ollama.complete(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM completion failed: {e}", model=model)
            raise
    
    async def complete_with_fallback(
        self,
        prompt: str,
        context: Optional[TaskContext] = None,
        fallback_model: Optional[str] = None
    ) -> str:
        """
        Complete with automatic fallback to simpler model on failure.
        
        Args:
            prompt: The prompt to complete
            context: Task context
            fallback_model: Specific fallback model
            
        Returns:
            Model response
        """
        try:
            return await self.complete(prompt, context)
        except Exception as e:
            logger.warning(f"Primary model failed, trying fallback: {e}")
            
            # Fallback to ollama client default
            return await self.ollama.complete(prompt)
    
    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get model name for a specific tier."""
        return self.model_config[tier]["model"]
    
    def update_model_config(self, tier: ModelTier, config: Dict[str, Any]):
        """Update model configuration for a tier."""
        self.model_config[tier].update(config)
