"""Chat API endpoint - main user interaction point."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional, List

from src.database.connection import get_db_yield
from src.database.models import ConversationMessage
from src.master_ai.brain import MasterAI

# Dependency to get master_ai instance (circular import avoidance handled in main)
# We will define a get_master_ai function that will be overridden/injected in main
async def get_master_ai_dep():
    from src.api.main import get_master_ai
    return get_master_ai()

router = APIRouter()

class ChatRequest(BaseModel):
    """Structured request for user messages."""
    message: str
    context: Optional[dict] = None  # Additional context for the request

class AgentInfo(BaseModel):
    """Information about an agent."""
    name: str
    status: str = "idle"
    risk_level: str = "low"
    capabilities: List[str] = Field(default_factory=list)
    last_task: Optional[str] = None

class SystemState(BaseModel):
    """Current system state for the dashboard."""
    autonomous_mode: bool = False
    active_agents: List[AgentInfo] = Field(default_factory=list)
    running_workflows: List[dict] = Field(default_factory=list)
    pending_approvals_count: int = 0

class ChatResponse(BaseModel):
    """
    Structured response mapping King AI's reasoning to the API.
    Includes both the text response and any automated actions or pending approvals.
    """
    type: str  # conversation, action, error
    response: str
    actions_taken: list[dict] = Field(default_factory=list)
    pending_approvals: list[dict] = Field(default_factory=list)
    system_state: Optional[SystemState] = None  # Current system state
    metadata: dict = Field(default_factory=dict)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    master_ai: MasterAI = Depends(get_master_ai_dep)
):
    """
    Main chat endpoint - Full orchestrator access.
    
    Send any message - King AI will determine if it's a conversation,
    command, or query and respond appropriately.
    
    Supports:
    - Natural language commands to agents
    - Workflow execution
    - Agent spawning and delegation
    - Business creation
    - Autonomous mode control
    """
    result = await master_ai.process_input(request.message)
    
    # Build system state for dashboard
    agents_list = master_ai.agent_router.list_agents()
    system_state = SystemState(
        autonomous_mode=master_ai.autonomous_mode,
        active_agents=[
            AgentInfo(
                name=a.get('name', 'unknown'),
                status=a.get('status', 'idle'),
                risk_level=a.get('risk_level', 'low'),
                capabilities=a.get('capabilities', [])
            ) for a in agents_list
        ],
        running_workflows=[],  # TODO: Get from workflow executor
        pending_approvals_count=len(result.pending_approvals)
    )
    
    return ChatResponse(
        type=result.type,
        response=result.response,
        actions_taken=[action.dict() for action in result.actions_taken],
        pending_approvals=result.pending_approvals,
        system_state=system_state,
        metadata=getattr(result, 'metadata', {})
    )

@router.get("/status")
async def get_status(master_ai: MasterAI = Depends(get_master_ai_dep)):
    """Get current orchestrator status."""
    agents_list = master_ai.agent_router.list_agents()
    return {
        "autonomous_mode": master_ai.autonomous_mode,
        "agents": agents_list,
        "conversation_history_length": len(master_ai._conversation_history),
    }

@router.get("/agents")
async def list_agents(master_ai: MasterAI = Depends(get_master_ai_dep)):
    """List all available agents with their capabilities."""
    agents_list = master_ai.agent_router.list_agents()
    return {
        "count": len(agents_list),
        "agents": agents_list
    }

@router.post("/delegate")
async def delegate_to_agent(
    agent_name: str,
    task: str,
    master_ai: MasterAI = Depends(get_master_ai_dep)
):
    """Directly delegate a task to a specific agent."""
    result = await master_ai._delegate_to_agent(agent_name, task)
    return result

@router.get("/history")
async def get_history(db = Depends(get_db_yield)):
    """Retrieve full conversation history."""
    from sqlalchemy import select
    result = await db.execute(
        select(ConversationMessage).order_by(ConversationMessage.created_at.asc())
    )
    messages = result.scalars().all()
    return [
        {"role": m.role, "content": m.content, "timestamp": m.created_at}
        for m in messages
    ]

@router.post("/autonomous/start")
async def start_autonomous(master_ai: MasterAI = Depends(get_master_ai_dep)):
    """Start autonomous mode (6h optimization loop)."""
    master_ai.autonomous_mode = True
    import asyncio
    asyncio.create_task(master_ai.run_autonomous_loop())
    return {"status": "Autonomous mode started"}

@router.post("/autonomous/stop")
async def stop_autonomous(master_ai: MasterAI = Depends(get_master_ai_dep)):
    """Stop autonomous mode."""
    master_ai.autonomous_mode = False
    return {"status": "Autonomous mode stopped"}
