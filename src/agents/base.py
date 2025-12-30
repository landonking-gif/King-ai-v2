"""Base class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any
from src.utils.ollama_client import OllamaClient
from config.settings import settings

class SubAgent(ABC):
    """
    Abstract Base Class for all specialized agents.
    Provides a standard interface for execution and LLM access.
    
    Each agent specializes in a specific domain (research, finance, etc.)
    and is called by the MasterAI to execute specific tasks.
    """
    
    name: str = "base"
    description: str = "Base agent"
    
    def __init__(self):
        """Initializes the agent with an Ollama client for local inference."""
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model  # Agents use same model as Master
        )
    
    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """
        Execute a task and return the result.
        
        Args:
            task: {
                "name": str,
                "description": str,
                "input": dict,  # Task-specific parameters
            }
            
        Returns:
            {
                "success": bool,
                "output": Any,  # Task-specific output
                "error": str | None,
                "metadata": dict
            }
        """
        pass
    
    async def _ask_llm(self, prompt: str) -> str:
        """Helper to query the LLM for agent-specific reasoning."""
        return await self.ollama.complete(prompt)
