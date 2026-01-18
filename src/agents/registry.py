"""
Enhanced Agent Registry.

Dynamic agent registration with capability declarations,
dependency injection, and runtime discovery.
Based on multi-agent-orchestration patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type
from uuid import uuid4

from src.utils.structured_logging import get_logger

logger = get_logger("agent_registry")


class AgentCapability(str, Enum):
    """Standard agent capabilities."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CONTENT_CREATION = "content_creation"
    PLANNING = "planning"
    FINANCE = "finance"
    LEGAL = "legal"
    MARKETING = "marketing"
    ORCHESTRATION = "orchestration"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    DATA_PROCESSING = "data_processing"


class AgentStatus(str, Enum):
    """Agent availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    DISABLED = "disabled"
    ERROR = "error"
    WARMING_UP = "warming_up"


@dataclass
class AgentMetadata:
    """Metadata describing an agent's capabilities."""
    id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: Set[AgentCapability] = field(default_factory=set)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    
    # Resource requirements
    requires_llm: bool = True
    requires_internet: bool = False
    requires_database: bool = False
    estimated_duration_ms: int = 1000
    estimated_cost: float = 0.0
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Configuration
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime
    status: AgentStatus = AgentStatus.AVAILABLE
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    
    # Registration
    registered_at: datetime = field(default_factory=datetime.utcnow)
    registered_by: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [c.value for c in self.capabilities],
            "input_types": self.input_types,
            "output_types": self.output_types,
            "requires_llm": self.requires_llm,
            "requires_internet": self.requires_internet,
            "requires_database": self.requires_database,
            "estimated_duration_ms": self.estimated_duration_ms,
            "estimated_cost": self.estimated_cost,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "avg_duration_ms": self.avg_duration_ms,
            "registered_at": self.registered_at.isoformat(),
        }


@dataclass
class AgentExecutionResult:
    """Result from agent execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentExecutor:
    """Base class for agent executors."""
    
    metadata: AgentMetadata
    
    async def execute(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> AgentExecutionResult:
        """Execute the agent's task."""
        raise NotImplementedError
    
    async def validate_input(
        self,
        task: Dict[str, Any],
    ) -> List[str]:
        """Validate task input. Returns list of errors."""
        return []
    
    async def health_check(self) -> bool:
        """Check if agent is healthy."""
        return True


class AgentRegistry:
    """
    Registry for dynamic agent management.
    
    Features:
    - Dynamic registration
    - Capability-based discovery
    - Dependency resolution
    - Health monitoring
    - Runtime statistics
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._agents: Dict[str, AgentExecutor] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._by_capability: Dict[AgentCapability, Set[str]] = {
            cap: set() for cap in AgentCapability
        }
        self._by_name: Dict[str, str] = {}  # name -> id mapping
    
    def register(
        self,
        executor: AgentExecutor,
        metadata: AgentMetadata = None,
    ) -> str:
        """
        Register an agent executor.
        
        Args:
            executor: The agent executor instance
            metadata: Optional metadata (uses executor.metadata if not provided)
            
        Returns:
            Agent ID
        """
        meta = metadata or getattr(executor, "metadata", None)
        if not meta:
            raise ValueError("Agent must have metadata")
        
        # Store executor and metadata
        self._agents[meta.id] = executor
        self._metadata[meta.id] = meta
        self._by_name[meta.name.lower()] = meta.id
        
        # Index by capabilities
        for cap in meta.capabilities:
            self._by_capability[cap].add(meta.id)
        
        logger.info(
            f"Registered agent: {meta.name} (id={meta.id})",
            capabilities=[c.value for c in meta.capabilities],
        )
        
        return meta.id
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if agent_id not in self._agents:
            return False
        
        meta = self._metadata[agent_id]
        
        # Remove from indices
        for cap in meta.capabilities:
            self._by_capability[cap].discard(agent_id)
        
        if meta.name.lower() in self._by_name:
            del self._by_name[meta.name.lower()]
        
        del self._agents[agent_id]
        del self._metadata[agent_id]
        
        logger.info(f"Unregistered agent: {meta.name}")
        return True
    
    def get(self, agent_id: str) -> Optional[AgentExecutor]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_by_name(self, name: str) -> Optional[AgentExecutor]:
        """Get an agent by name (case-insensitive)."""
        agent_id = self._by_name.get(name.lower())
        if agent_id:
            return self._agents.get(agent_id)
        return None
    
    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get metadata for an agent."""
        return self._metadata.get(agent_id)
    
    def find_by_capability(
        self,
        capability: AgentCapability,
        status: AgentStatus = None,
    ) -> List[AgentExecutor]:
        """
        Find agents with a specific capability.
        
        Args:
            capability: Required capability
            status: Optional status filter
            
        Returns:
            List of matching agents
        """
        agent_ids = self._by_capability.get(capability, set())
        agents = []
        
        for agent_id in agent_ids:
            meta = self._metadata[agent_id]
            if status is None or meta.status == status:
                agents.append(self._agents[agent_id])
        
        return agents
    
    def find_by_capabilities(
        self,
        capabilities: Set[AgentCapability],
        match_all: bool = True,
    ) -> List[AgentExecutor]:
        """
        Find agents with multiple capabilities.
        
        Args:
            capabilities: Required capabilities
            match_all: If True, agent must have ALL capabilities
            
        Returns:
            List of matching agents
        """
        if not capabilities:
            return list(self._agents.values())
        
        # Get candidate agents
        if match_all:
            # Intersection of all capability sets
            candidate_ids = None
            for cap in capabilities:
                cap_agents = self._by_capability.get(cap, set())
                if candidate_ids is None:
                    candidate_ids = cap_agents.copy()
                else:
                    candidate_ids &= cap_agents
            candidate_ids = candidate_ids or set()
        else:
            # Union of all capability sets
            candidate_ids = set()
            for cap in capabilities:
                candidate_ids |= self._by_capability.get(cap, set())
        
        return [self._agents[aid] for aid in candidate_ids]
    
    def find_for_task(
        self,
        task_type: str,
        required_capabilities: Set[AgentCapability] = None,
    ) -> Optional[AgentExecutor]:
        """
        Find best agent for a task.
        
        Args:
            task_type: Type of task
            required_capabilities: Optional capability requirements
            
        Returns:
            Best matching agent or None
        """
        # Map task types to capabilities
        task_capability_map = {
            "research": AgentCapability.RESEARCH,
            "analyze": AgentCapability.ANALYSIS,
            "code": AgentCapability.CODE_GENERATION,
            "review": AgentCapability.CODE_REVIEW,
            "content": AgentCapability.CONTENT_CREATION,
            "plan": AgentCapability.PLANNING,
            "finance": AgentCapability.FINANCE,
            "legal": AgentCapability.LEGAL,
            "marketing": AgentCapability.MARKETING,
        }
        
        # Determine required capability
        capabilities = required_capabilities or set()
        if task_type in task_capability_map:
            capabilities.add(task_capability_map[task_type])
        
        if not capabilities:
            return None
        
        # Find candidates
        candidates = self.find_by_capabilities(capabilities, match_all=True)
        
        # Filter by availability
        available = [
            agent for agent in candidates
            if self._metadata[agent.metadata.id].status == AgentStatus.AVAILABLE
        ]
        
        if not available:
            return None
        
        # Sort by performance (lowest avg duration, then by error rate)
        def score(agent):
            meta = self._metadata[agent.metadata.id]
            error_rate = meta.error_count / max(1, meta.execution_count)
            return (error_rate, meta.avg_duration_ms)
        
        available.sort(key=score)
        return available[0]
    
    async def execute(
        self,
        agent_id: str,
        task: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> AgentExecutionResult:
        """
        Execute an agent's task.
        
        Args:
            agent_id: Agent to execute
            task: Task data
            context: Execution context
            
        Returns:
            Execution result
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return AgentExecutionResult(
                success=False,
                error=f"Agent not found: {agent_id}",
            )
        
        meta = self._metadata[agent_id]
        
        # Check availability
        if meta.status not in [AgentStatus.AVAILABLE, AgentStatus.BUSY]:
            return AgentExecutionResult(
                success=False,
                error=f"Agent not available: {meta.status.value}",
            )
        
        # Validate input
        errors = await agent.validate_input(task)
        if errors:
            return AgentExecutionResult(
                success=False,
                error=f"Validation failed: {', '.join(errors)}",
            )
        
        # Execute
        start_time = datetime.utcnow()
        meta.status = AgentStatus.BUSY
        
        try:
            result = await agent.execute(task, context or {})
            
            # Update stats
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.duration_ms = duration_ms
            
            meta.execution_count += 1
            meta.last_execution = datetime.utcnow()
            
            # Update rolling average duration
            meta.avg_duration_ms = (
                (meta.avg_duration_ms * (meta.execution_count - 1) + duration_ms)
                / meta.execution_count
            )
            
            if not result.success:
                meta.error_count += 1
            
            meta.status = AgentStatus.AVAILABLE
            return result
            
        except Exception as e:
            meta.error_count += 1
            meta.status = AgentStatus.ERROR
            logger.error(f"Agent execution failed: {e}", agent_id=agent_id)
            
            return AgentExecutionResult(
                success=False,
                error=str(e),
            )
    
    def list_all(self) -> List[AgentMetadata]:
        """List all registered agents."""
        return list(self._metadata.values())
    
    def list_available(self) -> List[AgentMetadata]:
        """List available agents."""
        return [
            meta for meta in self._metadata.values()
            if meta.status == AgentStatus.AVAILABLE
        ]
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all agents."""
        results = {}
        for agent_id, agent in self._agents.items():
            try:
                results[agent_id] = await agent.health_check()
                if not results[agent_id]:
                    self._metadata[agent_id].status = AgentStatus.ERROR
            except Exception:
                results[agent_id] = False
                self._metadata[agent_id].status = AgentStatus.ERROR
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._agents)
        available = sum(1 for m in self._metadata.values() if m.status == AgentStatus.AVAILABLE)
        busy = sum(1 for m in self._metadata.values() if m.status == AgentStatus.BUSY)
        error = sum(1 for m in self._metadata.values() if m.status == AgentStatus.ERROR)
        
        total_executions = sum(m.execution_count for m in self._metadata.values())
        total_errors = sum(m.error_count for m in self._metadata.values())
        
        return {
            "total_agents": total,
            "available": available,
            "busy": busy,
            "error": error,
            "total_executions": total_executions,
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, total_executions),
            "capabilities": {
                cap.value: len(agents) 
                for cap, agents in self._by_capability.items()
            },
        }


# Decorator for easy agent registration
def agent(
    name: str,
    capabilities: List[AgentCapability],
    description: str = "",
    **kwargs,
):
    """
    Decorator to register an agent executor class.
    
    Usage:
        @agent("research", [AgentCapability.RESEARCH])
        class ResearchAgent(AgentExecutor):
            ...
    """
    def decorator(cls: Type[AgentExecutor]):
        cls.metadata = AgentMetadata(
            name=name,
            description=description or f"{name} agent",
            capabilities=set(capabilities),
            **kwargs,
        )
        return cls
    return decorator


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
