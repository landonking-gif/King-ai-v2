"""
Multi-Agent Orchestrator.
Coordinates execution across multiple specialized agents.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set, Awaitable
from enum import Enum
import uuid

from src.utils.structured_logging import get_logger

logger = get_logger("orchestrator")


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentCapability(str, Enum):
    """Agent capabilities."""
    RESEARCH = "research"
    CONTENT = "content"
    ANALYTICS = "analytics"
    COMMERCE = "commerce"
    LEGAL = "legal"
    FINANCE = "finance"
    CODE = "code"
    SUPPLIER = "supplier"
    BANKING = "banking"


@dataclass
class AgentInfo:
    """Information about an available agent."""
    id: str
    name: str
    capabilities: Set[AgentCapability]
    handler: Callable[[Dict], Awaitable[Dict]]
    max_concurrent: int = 3
    current_load: int = 0
    priority_multiplier: float = 1.0  # Lower = higher priority
    
    @property
    def available(self) -> bool:
        return self.current_load < self.max_concurrent
    
    @property
    def load_factor(self) -> float:
        return self.current_load / self.max_concurrent if self.max_concurrent > 0 else 1.0


@dataclass
class OrchestratorTask:
    """A task to be executed by agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    required_capability: Optional[AgentCapability] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A workflow of coordinated tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tasks: List[OrchestratorTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def completed_tasks(self) -> List[OrchestratorTask]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    @property
    def failed_tasks(self) -> List[OrchestratorTask]:
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]
    
    @property
    def pending_tasks(self) -> List[OrchestratorTask]:
        return [t for t in self.tasks if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)]
    
    @property
    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        return len(self.completed_tasks) / len(self.tasks)


@dataclass
class ExecutionResult:
    """Result of task/workflow execution."""
    success: bool
    task_id: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    retries_used: int = 0


class TaskScheduler:
    """Schedules tasks based on priority and dependencies."""
    
    def __init__(self):
        self._pending: List[OrchestratorTask] = []
        self._running: Dict[str, OrchestratorTask] = {}
    
    def add(self, task: OrchestratorTask) -> None:
        """Add task to scheduler."""
        task.status = TaskStatus.QUEUED
        self._pending.append(task)
        self._sort_pending()
    
    def _sort_pending(self) -> None:
        """Sort pending tasks by priority."""
        self._pending.sort(key=lambda t: (t.priority.value, t.created_at))
    
    def get_ready_tasks(
        self,
        completed_ids: Set[str],
        max_count: int = 10,
    ) -> List[OrchestratorTask]:
        """Get tasks that are ready to execute."""
        ready = []
        
        for task in self._pending:
            # Check dependencies
            deps_satisfied = all(dep in completed_ids for dep in task.dependencies)
            
            if deps_satisfied:
                ready.append(task)
                if len(ready) >= max_count:
                    break
        
        return ready
    
    def start_task(self, task: OrchestratorTask) -> None:
        """Mark task as running."""
        if task in self._pending:
            self._pending.remove(task)
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self._running[task.id] = task
    
    def complete_task(
        self,
        task_id: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark task as completed."""
        if task_id in self._running:
            task = self._running.pop(task_id)
            task.completed_at = datetime.utcnow()
            
            if error:
                task.status = TaskStatus.FAILED
                task.error = error
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
    
    @property
    def pending_count(self) -> int:
        return len(self._pending)
    
    @property
    def running_count(self) -> int:
        return len(self._running)


class AgentPool:
    """Manages pool of available agents."""
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
    
    def register(self, agent: AgentInfo) -> None:
        """Register an agent."""
        self._agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} with capabilities: {agent.capabilities}")
    
    def unregister(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_id, None)
    
    def find_agent(
        self,
        capability: Optional[AgentCapability] = None,
    ) -> Optional[AgentInfo]:
        """Find best available agent for capability."""
        candidates = []
        
        for agent in self._agents.values():
            if not agent.available:
                continue
            
            if capability and capability not in agent.capabilities:
                continue
            
            candidates.append(agent)
        
        if not candidates:
            return None
        
        # Sort by load factor and priority multiplier
        candidates.sort(key=lambda a: (a.load_factor, a.priority_multiplier))
        
        return candidates[0]
    
    def acquire(self, agent_id: str) -> bool:
        """Acquire an agent slot."""
        agent = self._agents.get(agent_id)
        if agent and agent.available:
            agent.current_load += 1
            return True
        return False
    
    def release(self, agent_id: str) -> None:
        """Release an agent slot."""
        agent = self._agents.get(agent_id)
        if agent and agent.current_load > 0:
            agent.current_load -= 1
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    @property
    def all_agents(self) -> List[AgentInfo]:
        return list(self._agents.values())
    
    @property
    def available_agents(self) -> List[AgentInfo]:
        return [a for a in self._agents.values() if a.available]


class MultiAgentOrchestrator:
    """
    Orchestrates execution across multiple specialized agents.
    
    Features:
    - Dynamic agent discovery and registration
    - Task scheduling with priorities and dependencies
    - Parallel execution with load balancing
    - Workflow management
    - Error handling and retries
    """
    
    def __init__(
        self,
        max_parallel_tasks: int = 10,
        default_timeout: float = 300.0,
    ):
        self.agent_pool = AgentPool()
        self.scheduler = TaskScheduler()
        self._max_parallel = max_parallel_tasks
        self._default_timeout = default_timeout
        
        self._workflows: Dict[str, Workflow] = {}
        self._completed_task_ids: Set[str] = set()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        self._event_handlers: Dict[str, List[Callable]] = {
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "workflow_completed": [],
        }
        
        self._running = False
        self._executor_task: Optional[asyncio.Task] = None
    
    def register_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        handler: Callable[[Dict], Awaitable[Dict]],
        max_concurrent: int = 3,
    ) -> None:
        """Register an agent with the orchestrator."""
        agent = AgentInfo(
            id=agent_id,
            name=name,
            capabilities=set(capabilities),
            handler=handler,
            max_concurrent=max_concurrent,
        )
        self.agent_pool.register(agent)
    
    def on_event(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
    
    def _emit(self, event: str, *args) -> None:
        """Emit event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(*args)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def submit_task(
        self,
        name: str,
        capability: AgentCapability,
        input_data: Dict,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Submit a task for execution."""
        task = OrchestratorTask(
            name=name,
            required_capability=capability,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or [],
            timeout_seconds=timeout or self._default_timeout,
            metadata=metadata or {},
        )
        
        self.scheduler.add(task)
        logger.info(f"Task submitted: {task.name} ({task.id})")
        
        return task.id
    
    async def create_workflow(
        self,
        name: str,
        tasks: List[Dict],
        context: Optional[Dict] = None,
    ) -> Workflow:
        """
        Create a workflow of coordinated tasks.
        
        Args:
            name: Workflow name
            tasks: List of task definitions with:
                - name: Task name
                - capability: Required agent capability
                - input: Input data
                - depends_on: Optional list of task names this depends on
            context: Shared workflow context
            
        Returns:
            Created workflow
        """
        workflow = Workflow(
            name=name,
            context=context or {},
        )
        
        # Build task name to ID mapping
        name_to_id: Dict[str, str] = {}
        
        for task_def in tasks:
            task = OrchestratorTask(
                name=task_def.get("name", "unnamed"),
                required_capability=AgentCapability(task_def["capability"]),
                input_data=task_def.get("input", {}),
                priority=TaskPriority(task_def.get("priority", 3)),
                timeout_seconds=task_def.get("timeout", self._default_timeout),
                metadata={"workflow_id": workflow.id},
            )
            
            name_to_id[task.name] = task.id
            workflow.tasks.append(task)
        
        # Resolve dependencies
        for i, task_def in enumerate(tasks):
            depends_on = task_def.get("depends_on", [])
            for dep_name in depends_on:
                if dep_name in name_to_id:
                    workflow.tasks[i].dependencies.append(name_to_id[dep_name])
        
        self._workflows[workflow.id] = workflow
        logger.info(f"Workflow created: {name} with {len(tasks)} tasks")
        
        return workflow
    
    async def run_workflow(self, workflow_id: str) -> Workflow:
        """Execute a workflow."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow.status = TaskStatus.RUNNING
        
        # Submit all tasks to scheduler
        for task in workflow.tasks:
            self.scheduler.add(task)
        
        # Wait for all tasks to complete
        while workflow.pending_tasks or any(
            t.status == TaskStatus.RUNNING for t in workflow.tasks
        ):
            await self._process_ready_tasks()
            await asyncio.sleep(0.1)
        
        # Update workflow status
        if workflow.failed_tasks:
            workflow.status = TaskStatus.FAILED
        else:
            workflow.status = TaskStatus.COMPLETED
        
        workflow.completed_at = datetime.utcnow()
        
        self._emit("workflow_completed", workflow)
        logger.info(
            f"Workflow completed: {workflow.name} "
            f"({len(workflow.completed_tasks)}/{len(workflow.tasks)} succeeded)"
        )
        
        return workflow
    
    async def start(self) -> None:
        """Start the orchestrator execution loop."""
        if self._running:
            return
        
        self._running = True
        self._executor_task = asyncio.create_task(self._execution_loop())
        logger.info("Orchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        
        # Cancel running tasks
        for task_id, async_task in list(self._running_tasks.items()):
            async_task.cancel()
            try:
                await async_task
            except asyncio.CancelledError:
                pass
        
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Orchestrator stopped")
    
    async def _execution_loop(self) -> None:
        """Main execution loop."""
        while self._running:
            try:
                await self._process_ready_tasks()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_ready_tasks(self) -> None:
        """Process tasks that are ready to execute."""
        # Calculate how many tasks we can start
        available_slots = self._max_parallel - len(self._running_tasks)
        
        if available_slots <= 0:
            return
        
        # Get ready tasks
        ready_tasks = self.scheduler.get_ready_tasks(
            self._completed_task_ids,
            max_count=available_slots,
        )
        
        for task in ready_tasks:
            # Find an agent
            agent = self.agent_pool.find_agent(task.required_capability)
            
            if not agent:
                logger.debug(f"No agent available for {task.required_capability}")
                continue
            
            # Acquire agent and start task
            if self.agent_pool.acquire(agent.id):
                task.assigned_agent = agent.id
                self.scheduler.start_task(task)
                
                async_task = asyncio.create_task(
                    self._execute_task(task, agent)
                )
                self._running_tasks[task.id] = async_task
                
                self._emit("task_started", task)
    
    async def _execute_task(
        self,
        task: OrchestratorTask,
        agent: AgentInfo,
    ) -> ExecutionResult:
        """Execute a single task."""
        start_time = datetime.utcnow()
        retries = 0
        
        while retries <= task.max_retries:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.handler(task.input_data),
                    timeout=task.timeout_seconds,
                )
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                self.scheduler.complete_task(task.id, result=result)
                self._completed_task_ids.add(task.id)
                self._running_tasks.pop(task.id, None)
                self.agent_pool.release(agent.id)
                
                self._emit("task_completed", task, result)
                
                logger.info(f"Task completed: {task.name} in {duration:.2f}s")
                
                return ExecutionResult(
                    success=True,
                    task_id=task.id,
                    result=result,
                    duration_seconds=duration,
                    retries_used=retries,
                )
                
            except asyncio.TimeoutError:
                retries += 1
                task.retry_count = retries
                logger.warning(f"Task timeout: {task.name} (retry {retries})")
                
                if retries > task.max_retries:
                    error = f"Timeout after {retries} retries"
                    self.scheduler.complete_task(task.id, error=error)
                    task.status = TaskStatus.TIMEOUT
                
            except Exception as e:
                retries += 1
                task.retry_count = retries
                logger.error(f"Task error: {task.name} - {e}")
                
                if retries > task.max_retries:
                    error = str(e)
                    self.scheduler.complete_task(task.id, error=error)
        
        # Task failed
        duration = (datetime.utcnow() - start_time).total_seconds()
        self._running_tasks.pop(task.id, None)
        self.agent_pool.release(agent.id)
        
        self._emit("task_failed", task, task.error)
        
        return ExecutionResult(
            success=False,
            task_id=task.id,
            error=task.error,
            duration_seconds=duration,
            retries_used=retries,
        )
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        # Check workflows
        for workflow in self._workflows.values():
            for task in workflow.tasks:
                if task.id == task_id:
                    return task.status
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "agents": {
                "total": len(self.agent_pool.all_agents),
                "available": len(self.agent_pool.available_agents),
            },
            "tasks": {
                "pending": self.scheduler.pending_count,
                "running": len(self._running_tasks),
                "completed": len(self._completed_task_ids),
            },
            "workflows": {
                "total": len(self._workflows),
                "completed": sum(
                    1 for w in self._workflows.values()
                    if w.status == TaskStatus.COMPLETED
                ),
            },
        }


# Global orchestrator instance
orchestrator = MultiAgentOrchestrator()


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get the global orchestrator instance."""
    return orchestrator


# Convenience function for creating agent registrations
def create_agent_handler(
    agent_class: type,
) -> Callable[[Dict], Awaitable[Dict]]:
    """Create a handler function from an agent class."""
    async def handler(input_data: Dict) -> Dict:
        agent = agent_class()
        return await agent.execute(input_data)
    
    return handler
