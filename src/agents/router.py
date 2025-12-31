"""Agent Router - Selects and executes the appropriate agent for each task."""

from typing import Optional
from src.agents.base import SubAgent
from src.agents.research import ResearchAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.code_reviewer import CodeReviewerAgent
from src.agents.content import ContentAgent
from src.agents.commerce import CommerceAgent
from src.agents.finance import FinanceAgent
from src.agents.banking import BankingAgent
from src.agents.analytics import AnalyticsAgent
from src.agents.legal import LegalAgent
from src.agents.supplier import SupplierAgent
from src.utils.structured_logging import get_logger

logger = get_logger("agent_router")


class AgentRouter:
    """
    The 'Dispatcher' of the system.
    Dynamically routes tasks to the correct specialized sub-agent.
    Supports dynamic registration and risk-based routing.
    """
    
    def __init__(self):
        """Registers all available agents in a dynamic map."""
        self.agents: dict[str, SubAgent] = {}
        self._risk_thresholds = {
            "low": ["research", "content", "analytics"],
            "medium": ["code_generator", "commerce", "supplier"],
            "high": ["finance", "banking", "legal", "code_reviewer"]
        }
        
        # Register default agents
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register all default agents."""
        default_agents = {
            "research": ResearchAgent(),
            "code_generator": CodeGeneratorAgent(),
            "code_reviewer": CodeReviewerAgent(),
            "content": ContentAgent(),
            "commerce": CommerceAgent(),
            "finance": FinanceAgent(),
            "banking": BankingAgent(),
            "analytics": AnalyticsAgent(),
            "legal": LegalAgent(),
            "supplier": SupplierAgent(),
        }
        
        for name, agent in default_agents.items():
            self.register_agent(name, agent)
    
    def register_agent(self, name: str, agent: SubAgent):
        """
        Register a new agent dynamically.
        
        Args:
            name: The name to register the agent under
            agent: The agent instance
        """
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: The name of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False
    
    def get_agent(self, name: str) -> Optional[SubAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def get_risk_level(self, agent_name: str) -> str:
        """
        Get the risk level associated with an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        for risk_level, agents in self._risk_thresholds.items():
            if agent_name in agents:
                return risk_level
        return "medium"  # Default to medium risk
    
    async def execute(self, task: dict) -> dict:
        """
        Route a task to the appropriate agent and execute it.
        
        Args:
            task: Must contain "agent" key specifying which agent to use
            
        Returns:
            Agent execution result
        """
        agent_name = task.get("agent")
        
        if not agent_name:
            return {"success": False, "error": "No agent specified in task"}
        
        agent = self.agents.get(agent_name)
        
        if not agent:
            # Check for aliases
            aliases = {
                "code": "code_generator",
                "generate_code": "code_generator",
                "review": "code_reviewer",
                "review_code": "code_reviewer",
                "search": "research",
                "market_research": "research",
                "blog": "content",
                "write": "content",
                "shop": "commerce",
                "store": "commerce",
                "payment": "finance",
                "stripe": "finance",
                "bank": "banking",
                "plaid": "banking",
                "metrics": "analytics",
                "kpi": "analytics",
                "compliance": "legal",
                "contract": "legal",
                "sourcing": "supplier",
                "dropship": "supplier",
            }
            
            resolved_name = aliases.get(agent_name)
            if resolved_name:
                agent = self.agents.get(resolved_name)
                logger.debug(f"Resolved agent alias: {agent_name} -> {resolved_name}")
        
        if not agent:
            return {"success": False, "error": f"Unknown agent: {agent_name}"}
        
        # Add risk level to result metadata
        risk_level = self.get_risk_level(agent_name)
        
        try:
            result = await agent.execute(task)
            result["_risk_level"] = risk_level
            result["_agent"] = agent_name
            return result
        except Exception as e:
            logger.error(f"Agent execution failed", agent=agent_name, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "_agent": agent_name,
                "_risk_level": risk_level
            }
    
    def list_agents(self) -> list[dict]:
        """List all available agents and their capabilities."""
        return [
            {
                "name": name,
                "description": getattr(agent, 'description', 'No description'),
                "risk_level": self.get_risk_level(name)
            }
            for name, agent in self.agents.items()
        ]
    
    def get_agents_by_risk(self, risk_level: str) -> list[str]:
        """Get all agents at a specific risk level."""
        return self._risk_thresholds.get(risk_level, [])
