"""
Scheduler API Routes - Manage scheduled autonomous tasks.

Provides endpoints for:
- Viewing scheduler status
- Listing scheduled tasks
- Enabling/disabling tasks
- Manually triggering task execution
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from src.services.scheduler import scheduler

router = APIRouter()


class TaskActionResponse(BaseModel):
    """Response for task actions."""
    success: bool
    message: str
    task_id: str


class SchedulerStatusResponse(BaseModel):
    """Scheduler status response."""
    running: bool
    check_interval_seconds: int
    total_tasks: int
    active_tasks: int
    total_executions: int
    tasks: List[Dict[str, Any]]


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status():
    """
    Get comprehensive scheduler status.
    
    Returns information about:
    - Whether scheduler is running
    - Number of registered tasks
    - Status of each task
    """
    return scheduler.get_status()


@router.get("/tasks")
async def list_tasks() -> List[Dict[str, Any]]:
    """
    List all scheduled tasks.
    
    Returns details for each task including:
    - Task ID and name
    - Frequency and next run time
    - Execution statistics
    - Error information if any
    """
    return scheduler.list_tasks()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get a specific task by ID."""
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return scheduler._task_to_dict(task)


@router.post("/tasks/{task_id}/enable", response_model=TaskActionResponse)
async def enable_task(task_id: str):
    """
    Enable a scheduled task.
    
    Re-enables a task that was disabled (manually or due to errors).
    Resets the error count.
    """
    if scheduler.enable_task(task_id):
        task = scheduler.get_task(task_id)
        return TaskActionResponse(
            success=True,
            message=f"Task '{task.name}' enabled",
            task_id=task_id
        )
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.post("/tasks/{task_id}/disable", response_model=TaskActionResponse)
async def disable_task(task_id: str):
    """
    Disable a scheduled task.
    
    Prevents the task from running until re-enabled.
    """
    if scheduler.disable_task(task_id):
        task = scheduler.get_task(task_id)
        return TaskActionResponse(
            success=True,
            message=f"Task '{task.name}' disabled",
            task_id=task_id
        )
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.post("/tasks/{task_id}/run", response_model=TaskActionResponse)
async def run_task_now(task_id: str):
    """
    Manually trigger a task to run immediately.
    
    Useful for testing or forcing an immediate execution
    without waiting for the scheduled time.
    """
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    success = await scheduler.run_task_now(task_id)
    if success:
        return TaskActionResponse(
            success=True,
            message=f"Task '{task.name}' executed",
            task_id=task_id
        )
    raise HTTPException(status_code=500, detail="Failed to execute task")
