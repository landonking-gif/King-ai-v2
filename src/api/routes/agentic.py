"""
Agentic Framework API Routes.

Provides API endpoints for:
- Agentic framework agent listing
- Ralph code agent task submission
- Workflow execution
- MCP tool invocation
- Memory service access
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.services.agentic_framework_bridge import get_agentic_bridge, AgenticFrameworkBridge
from src.utils.structured_logging import get_logger

logger = get_logger("agentic_api")
router = APIRouter()


def get_bridge() -> AgenticFrameworkBridge:
    """Get the agentic framework bridge instance."""
    return get_agentic_bridge()


# ============================================================================
# Request/Response Models
# ============================================================================

class RalphTaskRequest(BaseModel):
    """Request for Ralph code agent task."""
    task_description: str = Field(..., description="Description of the coding task")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    files_context: List[str] = Field(default_factory=list, description="Files to consider")
    target_server: Optional[str] = Field(None, description="Target server for execution")


class WorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    manifest_id: str = Field(..., description="ID of the workflow manifest")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input data")


class MCPToolRequest(BaseModel):
    """Request to invoke an MCP tool."""
    tool_name: str = Field(..., description="Name of the tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MemoryRequest(BaseModel):
    """Request to store/retrieve memory."""
    content: Optional[str] = Field(None, description="Content to store")
    query: Optional[str] = Field(None, description="Query for retrieval")
    memory_type: str = Field("short_term", description="Type of memory")
    limit: int = Field(10, description="Max results for retrieval")


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/agents")
async def list_agentic_agents(
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """List all available agentic framework agents."""
    agents = bridge.list_agentic_agents()
    return {
        "count": len(agents),
        "agents": agents,
        "source": "agentic_framework"
    }


@router.get("/health")
async def check_orchestrator_health(
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Check health of the agentic framework orchestrator."""
    health = await bridge.check_orchestrator_health()
    return health


@router.get("/manifests")
async def list_manifests(
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """List available workflow manifests."""
    manifests = bridge.get_available_manifests()
    return {
        "count": len(manifests),
        "manifests": manifests
    }


@router.post("/ralph/task")
async def submit_ralph_task(
    request: RalphTaskRequest,
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """
    Submit a coding task to the Ralph autonomous code agent.
    
    Ralph will:
    1. Analyze the task and codebase
    2. Generate an implementation plan
    3. Write code and tests
    4. Submit for review
    """
    logger.info(f"Ralph task submitted: {request.task_description[:100]}")
    
    result = await bridge.execute_ralph_task(
        task_description=request.task_description,
        requirements=request.requirements,
        files_context=request.files_context,
        target_server=request.target_server
    )
    
    return result


@router.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowRequest,
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Execute a YAML workflow from the agentic framework."""
    logger.info(f"Workflow execution requested: {request.manifest_id}")
    
    result = await bridge.execute_workflow(
        manifest_id=request.manifest_id,
        inputs=request.inputs
    )
    
    return result


@router.post("/mcp/invoke")
async def invoke_mcp_tool(
    request: MCPToolRequest,
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Invoke an MCP gateway tool."""
    logger.info(f"MCP tool invocation: {request.tool_name}")
    
    result = await bridge.invoke_mcp_tool(
        tool_name=request.tool_name,
        arguments=request.arguments
    )
    
    return result


@router.post("/memory/store")
async def store_memory(
    request: MemoryRequest,
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Store content in the memory service."""
    if not request.content:
        raise HTTPException(400, "Content is required for storage")
    
    result = await bridge.store_memory(
        content=request.content,
        memory_type=request.memory_type
    )
    
    return result


@router.post("/memory/retrieve")
async def retrieve_memory(
    request: MemoryRequest,
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Retrieve memories matching a query."""
    if not request.query:
        raise HTTPException(400, "Query is required for retrieval")
    
    result = await bridge.retrieve_memory(
        query=request.query,
        memory_type=request.memory_type,
        limit=request.limit
    )
    
    return result


@router.get("/status")
async def get_agentic_status(
    bridge: AgenticFrameworkBridge = Depends(get_bridge),
):
    """Get overall status of the agentic framework integration."""
    health = await bridge.check_orchestrator_health()
    agents = bridge.list_agentic_agents()
    manifests = bridge.get_available_manifests()
    
    return {
        "orchestrator": health,
        "agents_count": len(agents),
        "manifests_count": len(manifests),
        "urls": {
            "orchestrator": bridge.orchestrator_url,
            "mcp_gateway": bridge.mcp_gateway_url,
            "memory_service": bridge.memory_service_url
        }
    }
