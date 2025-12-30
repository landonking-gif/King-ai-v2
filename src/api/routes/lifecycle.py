"""
Lifecycle API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.business.lifecycle import EnhancedLifecycleEngine
from src.business.lifecycle_models import LifecycleStage, MilestoneType, TransitionTrigger
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])

_engine: Optional[EnhancedLifecycleEngine] = None


def get_engine() -> EnhancedLifecycleEngine:
    global _engine
    if _engine is None:
        _engine = EnhancedLifecycleEngine()
    return _engine


class InitRequest(BaseModel):
    business_id: str
    initial_stage: str = "ideation"


class TransitionRequest(BaseModel):
    business_id: str
    to_stage: str
    notes: str = ""
    force: bool = False


class MilestoneUpdateRequest(BaseModel):
    business_id: str
    milestone_id: str
    current_value: float


class AddMilestoneRequest(BaseModel):
    business_id: str
    name: str
    milestone_type: str
    target_value: float = Field(..., gt=0)
    target_date: Optional[str] = None


@router.post("/init")
async def initialize_business(req: InitRequest, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Initialize lifecycle tracking for a business."""
    try:
        stage = LifecycleStage(req.initial_stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {req.initial_stage}")
    
    state = await engine.initialize_business(req.business_id, stage)
    return {
        "business_id": state.business_id,
        "stage": state.current_stage.value,
        "milestones": len(state.milestones),
    }


@router.get("/state/{business_id}")
async def get_state(business_id: str, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Get current lifecycle state."""
    state = await engine.get_state(business_id)
    if not state:
        raise HTTPException(404, "Business not found")
    
    return {
        "business_id": state.business_id,
        "stage": state.current_stage.value,
        "entered_at": state.entered_at.isoformat(),
        "health_score": state.health_score,
        "next_stage": state.next_stage.value if state.next_stage else None,
        "blockers": state.blockers,
        "milestones": [
            {
                "id": m.id,
                "name": m.name,
                "type": m.milestone_type.value,
                "target": m.target_value,
                "current": m.current_value,
                "progress": m.progress_percent,
                "achieved": m.achieved,
                "overdue": m.is_overdue,
            }
            for m in state.milestones
        ],
        "transition_count": len(state.transitions),
    }


@router.post("/transition")
async def transition(req: TransitionRequest, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Transition to a new lifecycle stage."""
    try:
        to_stage = LifecycleStage(req.to_stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {req.to_stage}")
    
    success, message = await engine.transition(
        req.business_id,
        to_stage,
        TransitionTrigger.MANUAL,
        "api_user",
        req.notes,
        req.force,
    )
    
    if not success:
        raise HTTPException(400, message)
    
    return {"status": "ok", "message": message}


@router.post("/milestones/update")
async def update_milestone(req: MilestoneUpdateRequest, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Update milestone progress."""
    milestone = await engine.update_milestone(
        req.business_id, req.milestone_id, req.current_value
    )
    if not milestone:
        raise HTTPException(404, "Milestone not found")
    
    return {
        "milestone_id": milestone.id,
        "name": milestone.name,
        "progress": milestone.progress_percent,
        "achieved": milestone.achieved,
    }


@router.post("/milestones/add")
async def add_milestone(req: AddMilestoneRequest, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Add a custom milestone."""
    try:
        m_type = MilestoneType(req.milestone_type)
    except ValueError:
        raise HTTPException(400, f"Invalid type: {req.milestone_type}")
    
    milestone = await engine.add_milestone(
        req.business_id, req.name, m_type, req.target_value, req.target_date
    )
    if not milestone:
        raise HTTPException(404, "Business not found")
    
    return {"milestone_id": milestone.id, "name": milestone.name}


@router.get("/health/{business_id}")
async def get_health(business_id: str, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Calculate and return business health score."""
    score = await engine.calculate_health(business_id)
    return {"business_id": business_id, "health_score": score}


@router.get("/recommendations/{business_id}")
async def get_recommendations(business_id: str, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Get recommendations for business progress."""
    recommendations = await engine.get_recommendations(business_id)
    return {"recommendations": recommendations, "count": len(recommendations)}


@router.get("/stages")
async def list_stages(engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """List all lifecycle stages with info."""
    stages = [engine.get_stage_info(stage) for stage in LifecycleStage]
    return {"stages": stages}


@router.get("/stages/{stage}")
async def get_stage_info(stage: str, engine: EnhancedLifecycleEngine = Depends(get_engine)):
    """Get information about a specific stage."""
    try:
        lifecycle_stage = LifecycleStage(stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {stage}")
    
    return engine.get_stage_info(lifecycle_stage)
