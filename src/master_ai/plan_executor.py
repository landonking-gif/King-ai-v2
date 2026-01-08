"""
Plan Executor - Manages execution of plans with state tracking.
Handles task scheduling, approval gates, and failure recovery.
"""

import asyncio
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from uuid import uuid4

from src.master_ai.planning_models import (
    ExecutionPlan, PlanTask, TaskStatus, ReplanRequest
)
from src.master_ai.react_planner import ReActPlanner
from src.agents.router import AgentRouter
from src.database.connection import get_db, get_db_ctx
from src.database.models import Task as DBTask
from src.utils.structured_logging import get_logger
from src.utils.monitoring import monitor

logger = get_logger("plan_executor")


class PlanExecutor:
    """
    Executes plans with state management and failure handling.
    """
    
    def __init__(
        self,
        planner: ReActPlanner,
        agent_router: AgentRouter,
        on_approval_needed: Callable = None,
        on_task_complete: Callable = None,
        on_plan_complete: Callable = None
    ):
        """
        Initialize the executor.
        
        Args:
            planner: ReAct planner for replanning
            agent_router: Router to dispatch tasks to agents
            on_approval_needed: Callback when approval is needed
            on_task_complete: Callback when a task completes
            on_plan_complete: Callback when plan completes
        """
        self.planner = planner
        self.agent_router = agent_router
        self.on_approval_needed = on_approval_needed
        self.on_task_complete = on_task_complete
        self.on_plan_complete = on_plan_complete
        
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._paused_plans: Dict[str, ExecutionPlan] = {}
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        auto_continue: bool = True
    ) -> ExecutionPlan:
        """
        Execute a plan, handling approvals and failures.
        
        Args:
            plan: The plan to execute
            auto_continue: Whether to auto-execute ready tasks
            
        Returns:
            Updated plan with results
        """
        logger.info("Starting plan execution", plan_id=plan.id, tasks=len(plan.tasks))
        monitor.increment("plan_executor.plans_started")
        
        plan.status = "executing"
        plan.started_at = datetime.now()
        self._active_plans[plan.id] = plan
        
        try:
            while plan.status == "executing":
                # Get next ready task
                next_task = plan.get_next_task()
                
                if next_task is None:
                    # Check if we're done or blocked
                    if plan.completed_tasks == plan.total_tasks:
                        plan.status = "completed"
                        plan.completed_at = datetime.now()
                        break
                    
                    # Check for pending approvals
                    waiting = [t for t in plan.tasks if t.status == TaskStatus.WAITING_APPROVAL]
                    if waiting:
                        logger.info("Plan waiting for approvals", count=len(waiting))
                        plan.status = "paused"
                        self._paused_plans[plan.id] = plan
                        break
                    
                    # Check for failures blocking progress
                    failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
                    if failed:
                        logger.warning("Plan blocked by failures", count=len(failed))
                        plan.status = "failed"
                        break
                    
                    # Shouldn't reach here
                    logger.error("Plan in unexpected state")
                    break
                
                # Check if approval needed
                if next_task.requires_approval and next_task.status != TaskStatus.APPROVED:
                    next_task.status = TaskStatus.WAITING_APPROVAL
                    await self._request_approval(plan, next_task)
                    
                    if not auto_continue:
                        plan.status = "paused"
                        break
                    continue
                
                # Execute the task
                plan.current_task_id = next_task.id
                await self._execute_task(plan, next_task)
                
                plan.update_metrics()
                
                if self.on_task_complete:
                    await self.on_task_complete(plan, next_task)
            
            # Cleanup
            if plan.id in self._active_plans:
                del self._active_plans[plan.id]
            
            if self.on_plan_complete:
                await self.on_plan_complete(plan)
            
            monitor.increment(f"plan_executor.plans_{plan.status}")
            logger.info(
                "Plan execution finished",
                plan_id=plan.id,
                status=plan.status,
                completed=plan.completed_tasks,
                failed=plan.failed_tasks
            )
            
            return plan
            
        except Exception as e:
            logger.error("Plan execution error", error=str(e), exc_info=True)
            plan.status = "failed"
            raise
    
    async def _execute_task(self, plan: ExecutionPlan, task: PlanTask):
        """Execute a single task."""
        logger.info("Executing task", task_id=task.id, name=task.name, agent=task.agent)
        monitor.increment("plan_executor.tasks_started", tags={"agent": task.agent})
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Use ReAct loop for complex tasks, direct execution for simple ones
            if self._should_use_react(task):
                task = await self.planner.run_react_loop(
                    task,
                    self._execute_action
                )
            else:
                result = await self.agent_router.execute({
                    "name": task.name,
                    "description": task.description,
                    "agent": task.agent,
                    "input": task.input_data
                })
                
                task.output_data = result
                if result.get("success"):
                    task.status = TaskStatus.COMPLETED
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.get("error", "Unknown error")
                
                task.completed_at = datetime.now()
            
            # Persist to database
            await self._persist_task_result(task)
            
            monitor.increment(
                f"plan_executor.tasks_{task.status.value}",
                tags={"agent": task.agent}
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error("Task execution failed", task_id=task.id, error=str(e))
            monitor.increment("plan_executor.tasks_failed", tags={"agent": task.agent})
            
            # Attempt replanning if enabled
            await self._handle_failure(plan, task)
    
    def _should_use_react(self, task: PlanTask) -> bool:
        """Determine if a task should use the ReAct loop."""
        # Use ReAct for complex, multi-step tasks
        complex_agents = ["research", "code_generator", "analytics"]
        return task.agent in complex_agents and task.estimated_duration_minutes > 10
    
    async def _execute_action(
        self,
        task: PlanTask,
        action: str,
        action_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an action as part of ReAct loop."""
        return await self.agent_router.execute({
            "name": f"{task.name}:{action}",
            "description": task.description,
            "agent": task.agent,
            "input": {**task.input_data, **action_input}
        })
    
    async def _request_approval(self, plan: ExecutionPlan, task: PlanTask):
        """Request approval for a task."""
        logger.info(
            "Requesting approval",
            task_id=task.id,
            reason=task.approval_reason
        )
        
        # Persist to database for dashboard
        async with get_db_ctx() as db:
            db_task = DBTask(
                id=task.id,
                name=task.name,
                description=task.description,
                type=task.agent,
                status="pending_approval",
                agent=task.agent,
                input_data=task.input_data,
                requires_approval=True
            )
            db.add(db_task)
            await db.commit()
        
        if self.on_approval_needed:
            await self.on_approval_needed(plan, task)
    
    async def approve_task(self, plan_id: str, task_id: str, approver: str = "admin"):
        """Approve a pending task and resume execution."""
        plan = self._paused_plans.get(plan_id) or self._active_plans.get(plan_id)
        
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan")
        
        task.status = TaskStatus.APPROVED
        task.approved_by = approver
        task.approved_at = datetime.now()
        
        logger.info("Task approved", task_id=task_id, approver=approver)
        
        # Update database
        async with get_db_ctx() as db:
            db_task = await db.get(DBTask, task_id)
            if db_task:
                db_task.status = "approved"
                db_task.approved_by = approver
                db_task.approved_at = datetime.now()
                await db.commit()
        
        # Resume plan if it was paused
        if plan.status == "paused":
            plan.status = "executing"
            if plan.id in self._paused_plans:
                del self._paused_plans[plan.id]
            asyncio.create_task(self.execute_plan(plan))
    
    async def reject_task(self, plan_id: str, task_id: str, reason: str = None):
        """Reject a pending task."""
        plan = self._paused_plans.get(plan_id) or self._active_plans.get(plan_id)
        
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        task = plan.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan")
        
        task.status = TaskStatus.SKIPPED
        task.error = f"Rejected: {reason}" if reason else "Rejected by user"
        
        logger.info("Task rejected", task_id=task_id, reason=reason)
        
        # Update metrics and potentially fail the plan
        plan.update_metrics()
    
    async def _handle_failure(self, plan: ExecutionPlan, failed_task: PlanTask):
        """Handle task failure with potential replanning."""
        # Check if task has dependents that should be skipped
        for dependent_id in failed_task.blocks:
            dependent = plan.get_task(dependent_id)
            if dependent:
                dependent.status = TaskStatus.SKIPPED
                dependent.error = f"Skipped due to failure of dependency: {failed_task.name}"
        
        # TODO: Implement automatic replanning based on settings
        # For now, just log the failure
        logger.warning(
            "Task failed, dependents skipped",
            task_id=failed_task.id,
            dependents=len(failed_task.blocks)
        )
    
    async def _persist_task_result(self, task: PlanTask):
        """Save task result to database."""
        async with get_db_ctx() as db:
            db_task = await db.get(DBTask, task.id)
            if db_task:
                db_task.status = task.status.value
                db_task.output_data = task.output_data
                db_task.completed_at = task.completed_at
                await db.commit()
            else:
                # Create new record
                db_task = DBTask(
                    id=task.id,
                    name=task.name,
                    description=task.description,
                    type=task.agent,
                    status=task.status.value,
                    agent=task.agent,
                    input_data=task.input_data,
                    output_data=task.output_data,
                    requires_approval=task.requires_approval,
                    completed_at=task.completed_at
                )
                db.add(db_task)
                await db.commit()
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a plan."""
        plan = self._active_plans.get(plan_id) or self._paused_plans.get(plan_id)
        
        if not plan:
            return None
        
        return {
            "id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "progress": {
                "total": plan.total_tasks,
                "completed": plan.completed_tasks,
                "failed": plan.failed_tasks,
                "percentage": (plan.completed_tasks / plan.total_tasks * 100) if plan.total_tasks > 0 else 0
            },
            "current_task": plan.current_task_id,
            "pending_approvals": [
                {"id": t.id, "name": t.name, "reason": t.approval_reason}
                for t in plan.tasks if t.status == TaskStatus.WAITING_APPROVAL
            ]
        }
