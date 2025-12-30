"""
Playbook API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.business.playbook_models import PlaybookType, TriggerType
from src.business.playbook_loader import PlaybookLoader
from src.business.playbook_executor import PlaybookExecutor
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/playbooks", tags=["playbooks"])

_loader: Optional[PlaybookLoader] = None
_executor: Optional[PlaybookExecutor] = None


def get_loader() -> PlaybookLoader:
    global _loader
    if _loader is None:
        _loader = PlaybookLoader()
    return _loader


def get_executor() -> PlaybookExecutor:
    global _executor
    if _executor is None:
        _executor = PlaybookExecutor()
    return _executor


class ExecuteRequest(BaseModel):
    playbook_id: str
    business_id: str
    context: dict = {}


class ExecuteTemplateRequest(BaseModel):
    playbook_type: str
    business_id: str
    context: dict = {}


@router.get("/templates")
async def list_templates():
    """List available playbook templates."""
    return {
        "templates": [
            {"type": t.value, "name": t.value.replace("_", " ").title()}
            for t in PlaybookType
        ]
    }


@router.get("/templates/{playbook_type}")
async def get_template(playbook_type: str, loader: PlaybookLoader = Depends(get_loader)):
    """Get a playbook template."""
    try:
        pb_type = PlaybookType(playbook_type)
    except ValueError:
        raise HTTPException(400, f"Invalid type: {playbook_type}")
    
    playbook = loader.load_from_template(pb_type)
    if not playbook:
        raise HTTPException(404, "Template not found")
    
    return {
        "id": playbook.id,
        "name": playbook.name,
        "description": playbook.description,
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "agent": t.agent,
                "action": t.action,
                "dependencies": t.dependencies,
            }
            for t in playbook.tasks
        ],
    }


@router.get("/")
async def list_playbooks(loader: PlaybookLoader = Depends(get_loader)):
    """List all loaded playbooks."""
    playbooks = loader.load_all()
    return {
        "playbooks": [
            {
                "id": p.id,
                "name": p.name,
                "type": p.playbook_type.value,
                "task_count": len(p.tasks),
            }
            for p in playbooks
        ]
    }


@router.get("/{playbook_id}")
async def get_playbook(playbook_id: str, loader: PlaybookLoader = Depends(get_loader)):
    """Get a playbook by ID."""
    playbook = loader.get_cached(playbook_id)
    if not playbook:
        raise HTTPException(404, "Playbook not found")
    
    return {
        "id": playbook.id,
        "name": playbook.name,
        "type": playbook.playbook_type.value,
        "description": playbook.description,
        "version": playbook.version,
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "agent": t.agent,
                "action": t.action,
                "dependencies": t.dependencies,
            }
            for t in playbook.tasks
        ],
    }


@router.post("/execute")
async def execute_playbook(
    req: ExecuteRequest,
    loader: PlaybookLoader = Depends(get_loader),
    executor: PlaybookExecutor = Depends(get_executor),
):
    """Execute a playbook."""
    playbook = loader.get_cached(req.playbook_id)
    if not playbook:
        raise HTTPException(404, "Playbook not found")
    
    run = await executor.execute(playbook, req.business_id, req.context)
    
    return {
        "run_id": run.id,
        "status": run.status.value,
        "progress": run.progress_percent,
    }


@router.post("/execute/template")
async def execute_template(
    req: ExecuteTemplateRequest,
    loader: PlaybookLoader = Depends(get_loader),
    executor: PlaybookExecutor = Depends(get_executor),
):
    """Execute a playbook from template."""
    try:
        pb_type = PlaybookType(req.playbook_type)
    except ValueError:
        raise HTTPException(400, f"Invalid type: {req.playbook_type}")
    
    playbook = loader.load_from_template(pb_type)
    if not playbook:
        raise HTTPException(404, "Template not found")
    
    run = await executor.execute(playbook, req.business_id, req.context)
    
    return {
        "run_id": run.id,
        "status": run.status.value,
        "progress": run.progress_percent,
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str, executor: PlaybookExecutor = Depends(get_executor)):
    """Get playbook run status."""
    run = await executor.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    
    return {
        "id": run.id,
        "playbook_id": run.playbook_id,
        "business_id": run.business_id,
        "status": run.status.value,
        "progress": run.progress_percent,
        "started_at": run.started_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error,
        "tasks": {
            tid: {
                "status": t.status.value,
                "attempts": t.attempts,
                "error": t.error,
                "duration": t.duration_seconds,
            }
            for tid, t in run.task_executions.items()
        },
    }


@router.get("/runs/business/{business_id}")
async def get_business_runs(
    business_id: str, executor: PlaybookExecutor = Depends(get_executor)
):
    """Get all runs for a business."""
    runs = await executor.get_runs_for_business(business_id)
    return {
        "runs": [
            {
                "id": r.id,
                "playbook_id": r.playbook_id,
                "status": r.status.value,
                "progress": r.progress_percent,
                "started_at": r.started_at.isoformat(),
            }
            for r in runs
        ]
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, executor: PlaybookExecutor = Depends(get_executor)):
    """Cancel a running playbook."""
    success = await executor.cancel_run(run_id)
    if not success:
        raise HTTPException(400, "Cannot cancel run")
    return {"status": "cancelled"}
