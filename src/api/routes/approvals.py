"""
Approval Routes - Human-in-the-loop task management.
"""

from fastapi import APIRouter, HTTPException
from src.database.connection import get_db
from src.database.models import Task
from sqlalchemy import select
from datetime import datetime

router = APIRouter()

@router.get("/pending")
async def list_pending_approvals():
    """Returns all tasks currently waiting for human approval."""
    async with get_db() as db:
        result = await db.execute(
            select(Task).where(Task.status == "pending_approval").order_by(Task.created_at.desc())
        )
        return result.scalars().all()

@router.post("/{task_id}/approve")
async def approve_task(task_id: str, reviewer: str = "Admin"):
    """Approves a pending task for execution."""
    async with get_db() as db:
        task = await db.get(Task, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task.status = "approved"
        task.approved_at = datetime.now()
        task.approved_by = reviewer
        
        await db.commit()
        return {"status": "approved", "task_id": task_id}

@router.post("/{task_id}/reject")
async def reject_task(task_id: str, reason: str = None):
    """Rejects a pending task."""
    async with get_db() as db:
        task = await db.get(Task, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task.status = "rejected"
        task.output_data = {"rejection_reason": reason}
        
        await db.commit()
        return {"status": "rejected", "task_id": task_id}
