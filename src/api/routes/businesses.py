"""
Business Routes - CRUD operations for managed business units.
Includes file visibility and autonomous business management.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from src.database.connection import get_db, get_db_ctx
from src.database.models import BusinessUnit, BusinessStatus
from src.business.unit import BusinessManager
from src.services.autonomous_business_engine import get_autonomous_engine
from sqlalchemy import select
import os

router = APIRouter()
business_manager = BusinessManager()
autonomous_engine = get_autonomous_engine()


class CloneRequest(BaseModel):
    """Request model for cloning a business."""
    new_name: str
    new_niche: Optional[str] = None
    inherit_config: bool = True


@router.get("/")
async def list_businesses():
    """Returns all businesses in the empire portfolio."""
    async with get_db_ctx() as db:
        result = await db.execute(select(BusinessUnit).order_by(BusinessUnit.created_at.desc()))
        return result.scalars().all()

@router.get("/{business_id}")
async def get_business(business_id: str):
    """Returns detailed status for a single business unit."""
    async with get_db_ctx() as db:
        unit = await db.get(BusinessUnit, business_id)
        if not unit:
            raise HTTPException(status_code=404, detail="Business not found")
        return unit

@router.post("/{business_id}/clone")
async def clone_business(business_id: str, request: CloneRequest):
    """
    Clone a successful business unit to a new market/niche.
    
    This implements the REPLICATION lifecycle stage, allowing proven
    business models to be replicated efficiently.
    """
    clone = await business_manager.clone_business(
        source_id=business_id,
        new_name=request.new_name,
        new_niche=request.new_niche,
        inherit_config=request.inherit_config,
    )
    
    if not clone:
        raise HTTPException(status_code=404, detail="Source business not found")
    
    return {
        "status": "cloned",
        "clone_id": clone.id,
        "clone_name": clone.name,
        "source_id": business_id,
    }

@router.delete("/{business_id}")
async def delete_business(business_id: str):
    """Removes a business unit from the portfolio."""
    async with get_db_ctx() as db:
        unit = await db.get(BusinessUnit, business_id)
        if not unit:
            raise HTTPException(status_code=404, detail="Business not found")
        await db.delete(unit)
        await db.commit()
        return {"status": "deleted"}


# ============================================================================
# AUTONOMOUS BUSINESS ENGINE ENDPOINTS - File Visibility & Action Logs
# ============================================================================

class AutonomousBusinessRequest(BaseModel):
    """Request to create a business autonomously."""
    prompt: str


@router.post("/autonomous/create")
async def create_business_autonomous(request: AutonomousBusinessRequest):
    """
    Create a business using the Autonomous Business Engine.
    
    This executes the full lifecycle:
    1. Understanding - AI analyzes the business type
    2. Research - Market research, competitors, trends
    3. Planning - Create informed business plan
    4. Execution - Delegate to specialized agents
    5. Verification - Check all work is complete
    6. Monitoring - Set up continuous monitoring
    
    Returns the complete blueprint with file list.
    """
    try:
        blueprint = await autonomous_engine.create_business(request.prompt)
        files = autonomous_engine.get_business_files(blueprint.business_id)
        action_log = autonomous_engine.get_action_log(blueprint.business_id)
        
        return {
            "status": "created",
            "business_id": blueprint.business_id,
            "business_name": blueprint.business_name,
            "business_type": blueprint.business_type,
            "phase": blueprint.phase.value,
            "files_created": len(files),
            "tasks_completed": len(blueprint.completed_tasks),
            "tasks_total": len(blueprint.tasks),
            "files": files,
            "action_log": action_log[:50],  # Last 50 actions
            "tasks": [
                {
                    "id": t.task_id,
                    "name": t.name,
                    "agent_type": t.agent_type,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None
                }
                for t in blueprint.tasks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/files")
async def get_business_files(business_id: str):
    """
    Get all files created for a business.
    
    Returns a list of files with their paths, sizes, and existence status.
    This allows users to SEE what the AI has created.
    """
    files = autonomous_engine.get_business_files(business_id)
    
    if not files:
        # Check if business exists in legacy location
        base_path = os.path.join("businesses", business_id)
        if os.path.exists(base_path):
            # Scan for files manually
            files = []
            for root, _, filenames in os.walk(base_path):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, base_path)
                    files.append({
                        "path": rel_path,
                        "full_path": filepath,
                        "exists": True,
                        "size": os.path.getsize(filepath)
                    })
        else:
            raise HTTPException(status_code=404, detail="Business not found or no files created")
    
    return {
        "business_id": business_id,
        "file_count": len(files),
        "files": files
    }


@router.get("/{business_id}/files/{file_path:path}")
async def get_business_file_content(business_id: str, file_path: str):
    """
    Get the content of a specific file from a business.
    
    Args:
        business_id: The business ID
        file_path: The relative path to the file within the business folder
    
    Returns the file content as text or as a download for binary files.
    """
    full_path = os.path.join("businesses", business_id, file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Security check - prevent path traversal
    real_path = os.path.realpath(full_path)
    business_path = os.path.realpath(os.path.join("businesses", business_id))
    if not real_path.startswith(business_path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if it's a text file
    text_extensions = {'.md', '.txt', '.json', '.yaml', '.yml', '.py', '.js', '.html', '.css', '.csv'}
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() in text_extensions:
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return PlainTextResponse(content, media_type="text/plain")
        except UnicodeDecodeError:
            return FileResponse(full_path)
    else:
        return FileResponse(full_path)


@router.get("/{business_id}/actions")
async def get_business_actions(business_id: str, limit: int = 100):
    """
    Get the action log for a business.
    
    Shows everything the AI did to create and manage the business:
    - Phase transitions
    - Agent delegations
    - File creations
    - Verification results
    - Monitoring updates
    """
    action_log = autonomous_engine.get_action_log(business_id)
    
    if not action_log:
        raise HTTPException(status_code=404, detail="Business not found or no actions recorded")
    
    return {
        "business_id": business_id,
        "action_count": len(action_log),
        "actions": action_log[:limit]
    }


@router.get("/{business_id}/blueprint")
async def get_business_blueprint(business_id: str):
    """
    Get the complete business blueprint.
    
    Returns the full state of the autonomous business including:
    - Business details
    - Market research results
    - All tasks and their status
    - All files created
    - Current phase
    - Monitoring configuration
    """
    if business_id not in autonomous_engine.active_blueprints:
        raise HTTPException(status_code=404, detail="Business blueprint not found")
    
    blueprint = autonomous_engine.active_blueprints[business_id]
    files = autonomous_engine.get_business_files(business_id)
    
    return {
        "business_id": blueprint.business_id,
        "business_name": blueprint.business_name,
        "business_type": blueprint.business_type,
        "value_proposition": blueprint.value_proposition,
        "target_market": blueprint.target_market,
        "revenue_model": blueprint.revenue_model,
        "phase": blueprint.phase.value,
        "market_research": {
            "market_size": blueprint.market_research.market_size if blueprint.market_research else None,
            "competitors": blueprint.market_research.competitors if blueprint.market_research else [],
            "trends": blueprint.market_research.trends if blueprint.market_research else [],
            "opportunities": blueprint.market_research.opportunities if blueprint.market_research else [],
            "threats": blueprint.market_research.threats if blueprint.market_research else []
        } if blueprint.market_research else None,
        "tasks": [
            {
                "id": t.task_id,
                "name": t.name,
                "description": t.description,
                "agent_type": t.agent_type,
                "status": t.status.value,
                "result": t.result,
                "files_created": t.files_created,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None
            }
            for t in blueprint.tasks
        ],
        "completed_tasks": blueprint.completed_tasks,
        "failed_tasks": blueprint.failed_tasks,
        "files_created": len(files),
        "files": files,
        "kpis": blueprint.kpis,
        "monitoring_config": blueprint.monitoring_config,
        "created_at": blueprint.created_at.isoformat(),
        "updated_at": blueprint.updated_at.isoformat()
    }


@router.get("/{business_id}/status")
async def get_business_autonomous_status(business_id: str):
    """
    Get a quick status summary of an autonomous business.
    
    Returns the current phase, completion percentage, and any issues.
    """
    if business_id not in autonomous_engine.active_blueprints:
        # Check legacy
        base_path = os.path.join("businesses", business_id)
        if os.path.exists(base_path):
            file_count = sum(1 for _, _, files in os.walk(base_path) for _ in files)
            return {
                "business_id": business_id,
                "status": "legacy",
                "phase": "unknown",
                "file_count": file_count,
                "message": "This business was created before autonomous tracking was enabled"
            }
        raise HTTPException(status_code=404, detail="Business not found")
    
    blueprint = autonomous_engine.active_blueprints[business_id]
    total_tasks = len(blueprint.tasks)
    completed = len(blueprint.completed_tasks)
    failed = len(blueprint.failed_tasks)
    
    completion_pct = (completed / total_tasks * 100) if total_tasks > 0 else 0
    
    return {
        "business_id": business_id,
        "business_name": blueprint.business_name,
        "business_type": blueprint.business_type,
        "status": "active" if blueprint.phase.value != "complete" else "complete",
        "phase": blueprint.phase.value,
        "tasks": {
            "total": total_tasks,
            "completed": completed,
            "failed": failed,
            "pending": total_tasks - completed - failed
        },
        "completion_percentage": round(completion_pct, 1),
        "files_created": len(blueprint.files_created),
        "has_market_research": blueprint.market_research is not None,
        "monitoring_enabled": bool(blueprint.monitoring_config),
        "updated_at": blueprint.updated_at.isoformat()
    }


@router.get("/autonomous/list")
async def list_autonomous_businesses():
    """
    List all businesses created by the Autonomous Business Engine.
    
    Returns summary information for each active business.
    """
    businesses = []
    
    for business_id, blueprint in autonomous_engine.active_blueprints.items():
        businesses.append({
            "business_id": business_id,
            "business_name": blueprint.business_name,
            "business_type": blueprint.business_type,
            "phase": blueprint.phase.value,
            "tasks_total": len(blueprint.tasks),
            "tasks_completed": len(blueprint.completed_tasks),
            "files_created": len(blueprint.files_created),
            "created_at": blueprint.created_at.isoformat()
        })
    
    return {
        "count": len(businesses),
        "businesses": sorted(businesses, key=lambda x: x['created_at'], reverse=True)
    }
