"""Base class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any, Optional, ClassVar
from enum import Enum
from dataclasses import dataclass
from src.utils.ollama_client import OllamaClient
from config.settings import settings


class AgentCapability(str, Enum):
    """Agent capabilities."""
    RESEARCH = "research"
    WEB_SCRAPING = "web_scraping"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    CONTENT_CREATION = "content_creation"
    COMMERCE = "commerce"
    FINANCE = "finance"
    ANALYTICS = "analytics"
    LEGAL = "legal"


@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: str = ""
    metadata: Optional[dict] = None


class SubAgent(ABC):
    """
    Abstract Base Class for all specialized agents.
    Provides a standard interface for execution and LLM access.
    
    Each agent specializes in a specific domain (research, finance, etc.)
    and is called by the MasterAI to execute specific tasks.
    """
    
    name: str = "base"
    description: str = "Base agent"
    
    # Function calling schema for LLM integration
    # Override this in subclasses to define agent capabilities
    FUNCTION_SCHEMA: ClassVar[Optional[dict]] = None
    
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
    
    @classmethod
    def get_function_schema(cls) -> Optional[dict]:
        """
        Get the function calling schema for this agent.
        
        Returns JSON schema that can be used with LLM function calling
        to enable structured invocation of this agent.
        """
        return cls.FUNCTION_SCHEMA
    
    async def _ask_llm(self, prompt: str) -> str:
        """Helper to query the LLM for agent-specific reasoning."""
        return await self.ollama.complete(prompt)


class BaseAgent(SubAgent):
    """
    Base agent with enhanced initialization.
    Supports different initialization patterns.
    """
    
    def __init__(self, name: str, llm_client: Optional[OllamaClient] = None):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            llm_client: Optional LLM client (if None, creates default)
        """
        self.name = name
        if llm_client:
            self.llm = llm_client
            self.ollama = llm_client
        else:
            super().__init__()
            self.llm = self.ollama


def get_all_function_schemas() -> list[dict]:
    """
    Get function schemas from all registered agents.
    
    Returns a list of JSON function definitions suitable for
    LLM function calling APIs.
    """
    # This will be populated when agents are registered
    from src.agents.router import agent_router
    
    schemas = []
    # list_agents() returns list[dict], so iterate over registered agents directly
    for name, agent in agent_router.agents.items():
        if hasattr(agent, 'FUNCTION_SCHEMA') and agent.FUNCTION_SCHEMA:
            schemas.append(agent.FUNCTION_SCHEMA)
    
    return schemas

