"""
Agent Management API Routes.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.agents.registry import AgentRegistry, AgentMetadata, AgentStatus, AgentCapability
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


class SpawnAgentRequest(BaseModel):
    name: str
    capabilities: List[str]
    config: Optional[Dict[str, Any]] = None


class UpdateAgentRequest(BaseModel):
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_agents(
    capability: Optional[str] = None,
    status: Optional[str] = None,
    registry: AgentRegistry = Depends(get_registry),
):
    """List all registered agents with optional filtering."""
    agents = registry.list_agents()

    # Apply filters
    if capability:
        try:
            cap = AgentCapability(capability)
            agents = [a for a in agents if cap in a.capabilities]
        except ValueError:
            pass

    if status:
        try:
            stat = AgentStatus(status)
            agents = [a for a in agents if a.status == stat]
        except ValueError:
            pass

    return {
        "count": len(agents),
        "agents": [agent.to_dict() for agent in agents]
    }


@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_registry),
):
    """Get details for a specific agent."""
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    return agent.to_dict()


@router.post("/")
async def spawn_agent(
    req: SpawnAgentRequest,
    registry: AgentRegistry = Depends(get_registry),
):
    """Spawn a new agent instance."""
    try:
        # Convert string capabilities to enum
        capabilities = set()
        for cap_str in req.capabilities:
            try:
                capabilities.add(AgentCapability(cap_str))
            except ValueError:
                continue

        # Create agent metadata
        metadata = AgentMetadata(
            name=req.name,
            capabilities=capabilities,
            default_config=req.config or {},
        )

        # Register the agent
        agent_id = await registry.register_agent(metadata)

        return {
            "agent_id": agent_id,
            "status": "spawned",
            "message": f"Agent '{req.name}' spawned successfully"
        }

    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        raise HTTPException(500, f"Failed to spawn agent: {str(e)}")


@router.put("/{agent_id}")
async def update_agent(
    agent_id: str,
    req: UpdateAgentRequest,
    registry: AgentRegistry = Depends(get_registry),
):
    """Update agent status or configuration."""
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    try:
        if req.status:
            try:
                new_status = AgentStatus(req.status)
                agent.status = new_status
            except ValueError:
                raise HTTPException(400, f"Invalid status: {req.status}")

        if req.config:
            agent.default_config.update(req.config)

        return {
            "agent_id": agent_id,
            "status": "updated",
            "message": f"Agent '{agent.name}' updated successfully"
        }

    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}")
        raise HTTPException(500, f"Failed to update agent: {str(e)}")


@router.delete("/{agent_id}")
async def destroy_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_registry),
):
    """Destroy an agent instance."""
    try:
        success = await registry.unregister_agent(agent_id)
        if not success:
            raise HTTPException(404, "Agent not found")

        return {
            "agent_id": agent_id,
            "status": "destroyed",
            "message": "Agent destroyed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to destroy agent {agent_id}: {e}")
        raise HTTPException(500, f"Failed to destroy agent: {str(e)}")


@router.get("/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    registry: AgentRegistry = Depends(get_registry),
):
    """Get performance metrics for an agent."""
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Mock metrics for now - in real implementation, collect from agent executor
    metrics = {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "throughput": 120.5,
        "uptime_percentage": 99.8,
        "execution_count": agent.execution_count,
        "error_count": agent.error_count,
        "avg_duration_ms": agent.avg_duration_ms,
        "last_execution": agent.last_execution.isoformat() if agent.last_execution else None
    }

    return metrics


@router.get("/stats/summary")
async def get_agents_summary(
    registry: AgentRegistry = Depends(get_registry),
):
    """Get summary statistics for all agents."""
    agents = registry.list_agents()

    total_agents = len(agents)
    running_agents = len([a for a in agents if a.status == AgentStatus.AVAILABLE])
    busy_agents = len([a for a in agents if a.status == AgentStatus.BUSY])
    error_agents = len([a for a in agents if a.status == AgentStatus.ERROR])

    capabilities_count = {}
    for agent in agents:
        for cap in agent.capabilities:
            capabilities_count[cap.value] = capabilities_count.get(cap.value, 0) + 1

    return {
        "total_agents": total_agents,
        "running_agents": running_agents,
        "busy_agents": busy_agents,
        "error_agents": error_agents,
        "capabilities_distribution": capabilities_count
    }