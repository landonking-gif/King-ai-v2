"""Chat API endpoint - main user interaction point."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any

from src.database.connection import get_db
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

class ChatResponse(BaseModel):
    """
    Structured response mapping King AI's reasoning to the API.
    Includes both the text response and any automated actions or pending approvals.
    """
    type: str  # conversation, action
    response: str
    actions_taken: list[dict]
    pending_approvals: list[dict]

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    master_ai: MasterAI = Depends(get_master_ai_dep)
):
    """
    Main chat endpoint.
    
    Send any message - King AI will determine if it's a conversation,
    command, or query and respond appropriately.
    """
    result = await master_ai.process_input(request.message)
    return ChatResponse(
        type=result.type,
        response=result.response,
        actions_taken=[action.dict() for action in result.actions_taken],
        pending_approvals=result.pending_approvals
    )

@router.get("/history")
async def get_history(db = Depends(get_db)):
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
