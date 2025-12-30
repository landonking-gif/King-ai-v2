"""Agent Router - Selects and executes the appropriate agent for each task."""

from src.agents.base import SubAgent
from src.agents.research import ResearchAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.content import ContentAgent
from src.agents.commerce import CommerceAgent
from src.agents.finance import FinanceAgent
from src.agents.analytics import AnalyticsAgent
from src.agents.legal import LegalAgent

class AgentRouter:
    """
    The 'Dispatcher' of the system.
    Dynamically routes tasks to the correct specialized sub-agent.
    """
    
    def __init__(self):
        """Registers all available agents in a dynamic map."""
        self.agents: dict[str, SubAgent] = {
            "research": ResearchAgent(),
            "code_generator": CodeGeneratorAgent(),
            "content": ContentAgent(),
            "commerce": CommerceAgent(),
            "finance": FinanceAgent(),
            "analytics": AnalyticsAgent(),
            "legal": LegalAgent(),
        }
    
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
            # Fallback for now if agent not found or for testing
            return {"success": False, "error": f"Unknown agent: {agent_name}"}
        
        return await agent.execute(task)
    
    def list_agents(self) -> list[dict]:
        """List all available agents and their capabilities."""
        return [
            {"name": name, "description": agent.description}
            for name, agent in self.agents.items()
        ]
