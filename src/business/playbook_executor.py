"""
Playbook Executor - Run playbooks and manage task execution.
"""
import asyncio
import uuid
from datetime import datetime, UTC
from typing import Any, Optional, Callable
from src.business.playbook_models import (
    PlaybookDefinition, PlaybookRun, TaskDefinition, TaskExecution,
    TaskStatus, TriggerType
)
from src.utils.structured_logging import get_logger

logger = get_logger(__name__)


class PlaybookExecutor:
    """Execute playbooks and orchestrate tasks."""

    def __init__(self):
        self._runs: dict[str, PlaybookRun] = {}
        self._agents: dict[str, Any] = {}  # Agent instances
        self._hooks: dict[str, list[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "playbook_completed": [],
        }

    def register_agent(self, name: str, agent: Any):
        """Register an agent for task execution."""
        self._agents[name] = agent

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for execution events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def execute(
        self,
        playbook: PlaybookDefinition,
        business_id: str,
        context: dict = None,
        trigger: TriggerType = TriggerType.MANUAL,
    ) -> PlaybookRun:
        """Execute a playbook."""
        run = PlaybookRun(
            id=str(uuid.uuid4()),
            playbook_id=playbook.id,
            business_id=business_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(UTC),
            context=context or {},
            triggered_by=trigger,
        )

        # Initialize task executions
        for task in playbook.tasks:
            run.task_executions[task.id] = TaskExecution(
                task_id=task.id,
                status=TaskStatus.PENDING,
            )

        self._runs[run.id] = run
        logger.info(f"Starting playbook run {run.id} for {playbook.name}")

        try:
            await self._execute_tasks(playbook, run)
            
            if run.failed_tasks:
                run.status = TaskStatus.FAILED
                run.error = f"Failed tasks: {', '.join(run.failed_tasks)}"
            else:
                run.status = TaskStatus.COMPLETED

        except Exception as e:
            run.status = TaskStatus.FAILED
            run.error = str(e)
            logger.error(f"Playbook execution error: {e}")

        run.completed_at = datetime.now(UTC)

        # Execute playbook completed hooks
        for hook in self._hooks["playbook_completed"]:
            try:
                await hook(run)
            except Exception as e:
                logger.error(f"Playbook hook error: {e}")

        return run

    async def _execute_tasks(self, playbook: PlaybookDefinition, run: PlaybookRun):
        """Execute tasks respecting dependencies."""
        completed = set()
        failed = set()

        while True:
            # Find tasks ready to run
            ready_tasks = []
            for task in playbook.tasks:
                execution = run.task_executions[task.id]
                
                if execution.status != TaskStatus.PENDING:
                    continue

                # Check dependencies
                deps_met = all(d in completed for d in task.dependencies)
                deps_failed = any(d in failed for d in task.dependencies)

                if deps_failed:
                    execution.status = TaskStatus.BLOCKED
                    continue

                if deps_met:
                    # Check conditions
                    if self._evaluate_conditions(task.conditions, run.context):
                        ready_tasks.append(task)
                    else:
                        execution.status = TaskStatus.SKIPPED
                        completed.add(task.id)

            if not ready_tasks:
                # Check if we're done or stuck
                pending = [
                    t for t in playbook.tasks
                    if run.task_executions[t.id].status == TaskStatus.PENDING
                ]
                if not pending:
                    break
                
                blocked = [
                    t for t in playbook.tasks
                    if run.task_executions[t.id].status == TaskStatus.BLOCKED
                ]
                if len(blocked) == len(pending):
                    break  # All remaining tasks are blocked

                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks concurrently
            tasks_coros = [
                self._execute_task(task, run)
                for task in ready_tasks
            ]
            results = await asyncio.gather(*tasks_coros, return_exceptions=True)

            for task, result in zip(ready_tasks, results):
                execution = run.task_executions[task.id]
                
                if isinstance(result, Exception):
                    execution.status = TaskStatus.FAILED
                    execution.error = str(result)
                    failed.add(task.id)
                elif execution.status == TaskStatus.COMPLETED:
                    completed.add(task.id)
                elif execution.status == TaskStatus.FAILED:
                    failed.add(task.id)

    async def _execute_task(
        self, task: TaskDefinition, run: PlaybookRun
    ) -> Any:
        """Execute a single task."""
        execution = run.task_executions[task.id]
        execution.status = TaskStatus.RUNNING
        execution.started_at = datetime.now(UTC)
        execution.attempts += 1

        # Execute task started hooks
        for hook in self._hooks["task_started"]:
            try:
                await hook(run.id, task.id)
            except Exception as e:
                logger.error(f"Task hook error: {e}")

        agent = self._agents.get(task.agent)
        if not agent:
            execution.status = TaskStatus.FAILED
            execution.error = f"Agent not found: {task.agent}"
            execution.completed_at = datetime.now(UTC)
            return None

        try:
            # Merge playbook context with task parameters
            context = {
                **run.context,
                "business_id": run.business_id,
                "action": task.action,
                **task.parameters,
            }

            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(task.action, context),
                timeout=task.timeout_seconds,
            )

            execution.result = result
            execution.status = TaskStatus.COMPLETED
            execution.completed_at = datetime.now(UTC)

            # Store result in context for dependent tasks
            run.context[f"{task.id}_result"] = result

            # Execute task completed hooks
            for hook in self._hooks["task_completed"]:
                try:
                    await hook(run.id, task.id, result)
                except Exception as e:
                    logger.error(f"Task hook error: {e}")

            return result

        except asyncio.TimeoutError:
            execution.status = TaskStatus.FAILED
            execution.error = "Task timed out"
            execution.completed_at = datetime.now(UTC)
            
            # Retry if attempts remaining
            if execution.attempts < task.retry_count:
                execution.status = TaskStatus.PENDING
                return await self._execute_task(task, run)

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(UTC)

            # Retry if attempts remaining
            if execution.attempts < task.retry_count:
                execution.status = TaskStatus.PENDING
                await asyncio.sleep(1)  # Brief delay before retry
                return await self._execute_task(task, run)

            # Execute task failed hooks
            for hook in self._hooks["task_failed"]:
                try:
                    await hook(run.id, task.id, str(e))
                except Exception as he:
                    logger.error(f"Task hook error: {he}")

        return None

    def _evaluate_conditions(self, conditions: list[dict], context: dict) -> bool:
        """Evaluate task conditions."""
        if not conditions:
            return True

        for cond in conditions:
            field = cond.get("field", "")
            operator = cond.get("operator", "eq")
            value = cond.get("value")

            actual = context.get(field)
            
            if operator == "eq" and actual != value:
                return False
            elif operator == "neq" and actual == value:
                return False
            elif operator == "exists" and field not in context:
                return False
            elif operator == "gt" and not (actual and actual > value):
                return False
            elif operator == "lt" and not (actual and actual < value):
                return False

        return True

    async def get_run(self, run_id: str) -> Optional[PlaybookRun]:
        """Get a playbook run by ID."""
        return self._runs.get(run_id)

    async def get_runs_for_business(self, business_id: str) -> list[PlaybookRun]:
        """Get all runs for a business."""
        return [r for r in self._runs.values() if r.business_id == business_id]

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running playbook."""
        run = self._runs.get(run_id)
        if not run or run.status != TaskStatus.RUNNING:
            return False

        run.status = TaskStatus.FAILED
        run.error = "Cancelled by user"
        run.completed_at = datetime.now(UTC)
        return True
