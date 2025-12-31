"""
Scheduler Service - Manages cron-based autonomous operations.
Implements scheduled tasks for the Master AI brain.

This enables true autonomous operation by running periodic tasks like:
- Business optimization loops (every 6 hours)
- KPI health checks (hourly)
- Evolution proposals (daily)
- Business unit reviews (daily)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.utils.logging import get_logger
from config.settings import settings

logger = get_logger("scheduler")


class TaskFrequency(str, Enum):
    """Predefined task frequencies."""
    EVERY_MINUTE = "every_minute"  # For testing
    EVERY_5_MINUTES = "every_5_minutes"
    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    DAILY = "daily"
    WEEKLY = "weekly"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    name: str
    frequency: TaskFrequency
    callback: Callable
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    error_count: int = 0
    max_errors: int = 5
    status: TaskStatus = TaskStatus.PENDING
    last_error: Optional[str] = None
    run_count: int = 0
    total_duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """
    Async scheduler for autonomous operations.
    
    Manages scheduled tasks for the King AI system, enabling true
    autonomous operation as specified in the design document.
    
    Key Features:
    - Configurable task frequencies
    - Error handling with automatic disable after max errors
    - Task status tracking and metrics
    - Graceful shutdown
    """
    
    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_interval = 30  # Check every 30 seconds
        
        # Frequency intervals in seconds
        self._intervals = {
            TaskFrequency.EVERY_MINUTE: 60,
            TaskFrequency.EVERY_5_MINUTES: 5 * 60,
            TaskFrequency.HOURLY: 3600,
            TaskFrequency.EVERY_6_HOURS: 6 * 3600,
            TaskFrequency.DAILY: 24 * 3600,
            TaskFrequency.WEEKLY: 7 * 24 * 3600,
        }
    
    def register_task(
        self,
        name: str,
        callback: Callable,
        frequency: TaskFrequency,
        enabled: bool = True,
        run_immediately: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a scheduled task.
        
        Args:
            name: Human-readable task name
            callback: Async function to call
            frequency: How often to run
            enabled: Whether task is active
            run_immediately: If True, run on first scheduler tick
            metadata: Optional metadata for the task
            
        Returns:
            Task ID (UUID string)
        """
        task_id = str(uuid.uuid4())
        
        # Calculate first run time
        if run_immediately:
            next_run = datetime.utcnow()
        else:
            interval = self._intervals.get(frequency, 3600)
            next_run = datetime.utcnow() + timedelta(seconds=interval)
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            frequency=frequency,
            callback=callback,
            enabled=enabled,
            next_run=next_run,
            metadata=metadata or {}
        )
        
        self._tasks[task_id] = task
        logger.info(
            f"Registered scheduled task: {name}",
            task_id=task_id,
            frequency=frequency.value,
            next_run=next_run.isoformat()
        )
        
        return task_id
    
    def unregister_task(self, task_id: str) -> bool:
        """Unregister a task by ID."""
        if task_id in self._tasks:
            task = self._tasks.pop(task_id)
            logger.info(f"Unregistered task: {task.name}", task_id=task_id)
            return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            self._tasks[task_id].status = TaskStatus.PENDING
            self._tasks[task_id].error_count = 0
            logger.info(f"Enabled task: {self._tasks[task_id].name}")
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            self._tasks[task_id].status = TaskStatus.DISABLED
            logger.info(f"Disabled task: {self._tasks[task_id].name}")
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_task_by_name(self, name: str) -> Optional[ScheduledTask]:
        """Get a task by name."""
        for task in self._tasks.values():
            if task.name == name:
                return task
        return None
    
    async def start(self):
        """Start the scheduler background loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Scheduler started",
            task_count=len(self._tasks),
            check_interval=self._check_interval
        )
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def run_task_now(self, task_id: str) -> bool:
        """Manually trigger a task to run immediately."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        await self._execute_task(task)
        return True
    
    async def _run_loop(self):
        """Main scheduler loop - checks and executes due tasks."""
        logger.info("Scheduler loop started")
        
        while self._running:
            try:
                now = datetime.utcnow()
                
                for task in list(self._tasks.values()):
                    if not task.enabled:
                        continue
                    
                    if task.status == TaskStatus.RUNNING:
                        continue  # Already running
                    
                    if task.next_run and now >= task.next_run:
                        # Run task in background to not block scheduler
                        asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
            
            # Wait before next check
            await asyncio.sleep(self._check_interval)
        
        logger.info("Scheduler loop ended")
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task with error handling and metrics."""
        task.status = TaskStatus.RUNNING
        start_time = datetime.utcnow()
        
        logger.info(f"Executing scheduled task: {task.name}", task_id=task.id)
        
        try:
            # Execute the callback
            if asyncio.iscoroutinefunction(task.callback):
                await task.callback()
            else:
                task.callback()
            
            # Record success
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            task.last_run = end_time
            task.run_count += 1
            task.total_duration_ms += duration_ms
            task.error_count = 0
            task.last_error = None
            task.status = TaskStatus.COMPLETED
            
            # Calculate next run
            interval = self._intervals.get(task.frequency, 3600)
            task.next_run = end_time + timedelta(seconds=interval)
            
            logger.info(
                f"Task {task.name} completed",
                task_id=task.id,
                duration_ms=round(duration_ms, 2),
                next_run=task.next_run.isoformat()
            )
            
        except Exception as e:
            # Record failure
            task.error_count += 1
            task.last_error = str(e)
            task.status = TaskStatus.FAILED
            
            logger.error(
                f"Task {task.name} failed",
                task_id=task.id,
                error=str(e),
                error_count=task.error_count,
                exc_info=True
            )
            
            # Disable after too many errors
            if task.error_count >= task.max_errors:
                task.enabled = False
                task.status = TaskStatus.DISABLED
                logger.warning(
                    f"Task {task.name} disabled after {task.max_errors} consecutive errors",
                    task_id=task.id
                )
            else:
                # Retry in 5 minutes
                task.next_run = datetime.utcnow() + timedelta(minutes=5)
                task.status = TaskStatus.PENDING
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        active_tasks = sum(1 for t in self._tasks.values() if t.enabled)
        total_runs = sum(t.run_count for t in self._tasks.values())
        
        return {
            "running": self._running,
            "check_interval_seconds": self._check_interval,
            "total_tasks": len(self._tasks),
            "active_tasks": active_tasks,
            "total_executions": total_runs,
            "tasks": [self._task_to_dict(t) for t in self._tasks.values()]
        }
    
    def _task_to_dict(self, task: ScheduledTask) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        avg_duration = (
            task.total_duration_ms / task.run_count 
            if task.run_count > 0 else 0
        )
        
        return {
            "id": task.id,
            "name": task.name,
            "frequency": task.frequency.value,
            "enabled": task.enabled,
            "status": task.status.value,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "error_count": task.error_count,
            "last_error": task.last_error,
            "avg_duration_ms": round(avg_duration, 2),
            "metadata": task.metadata
        }
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks as dictionaries."""
        return [self._task_to_dict(t) for t in self._tasks.values()]


# Global scheduler instance
scheduler = Scheduler()
