"""
Base Agent Class

This module provides the abstract base class for all agents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    tool: str
    tool_input: Dict[str, Any]
    log: str


@dataclass
class AgentFinish:
    """Represents the final output of an agent"""
    output: str
    log: str


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, description: str, llm: Any, tools: List[Any]):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = []
    
    @abstractmethod
    def plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> List[AgentAction]:
        """Plan actions to complete the task"""
        pass
    
    @abstractmethod
    def execute(self, action: AgentAction) -> str:
        """Execute a single action"""
        pass
    
    @abstractmethod
    def run(self, task: str) -> str:
        """Run the agent on a task"""
        pass
    
    def add_to_memory(self, interaction: Dict[str, Any]) -> None:
        """Add interaction to agent memory"""
        self.memory.append({
            **interaction,
            'timestamp': time.time()
        })
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List available tool names"""
        return list(self.tools.keys())
