# Implementation Plan Part 20: Business Playbook System

| Field | Value |
|-------|-------|
| Module | Business Playbook & Automation System |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 3 (Database), Part 19 (Lifecycle), Part 4 (Business Unit) |

---

## 1. Scope

This module implements the playbook system for business automation:

- **Playbook Definitions** - YAML-based business operation playbooks
- **Task Orchestration** - Execute multi-step workflows
- **Conditional Logic** - Branch execution based on conditions
- **Scheduling** - Time-based and event-based triggers
- **Progress Tracking** - Monitor playbook execution status

---

## 2. Tasks

### Task 20.1: Playbook Models

**File: `src/business/playbook_models.py`**

```python
"""
Playbook Data Models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable


class PlaybookType(Enum):
    """Types of business playbooks."""
    DROPSHIPPING = "dropshipping"
    SAAS = "saas"
    CONTENT = "content"
    ECOMMERCE = "ecommerce"
    SERVICE = "service"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TriggerType(Enum):
    """What triggers a playbook or task."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    CONDITION = "condition"
    WEBHOOK = "webhook"


@dataclass
class TaskDefinition:
    """Definition of a single task in a playbook."""
    id: str
    name: str
    description: str
    agent: str  # Which agent executes this
    action: str  # Action to perform
    parameters: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # Task IDs this depends on
    conditions: list[dict] = field(default_factory=list)  # Conditions to run
    timeout_seconds: int = 300
    retry_count: int = 3
    on_failure: str = "continue"  # continue, stop, skip_dependents


@dataclass
class TaskExecution:
    """Record of a task execution."""
    task_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


@dataclass
class PlaybookDefinition:
    """Definition of a business playbook."""
    id: str
    name: str
    playbook_type: PlaybookType
    description: str
    version: str = "1.0"
    tasks: list[TaskDefinition] = field(default_factory=list)
    triggers: list[dict] = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None


@dataclass
class PlaybookRun:
    """A single execution run of a playbook."""
    id: str
    playbook_id: str
    business_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    task_executions: dict[str, TaskExecution] = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    triggered_by: TriggerType = TriggerType.MANUAL
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if not self.task_executions:
            return 0.0
        completed = sum(
            1 for t in self.task_executions.values()
            if t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
        )
        return (completed / len(self.task_executions)) * 100

    @property
    def failed_tasks(self) -> list[str]:
        return [
            tid for tid, t in self.task_executions.items()
            if t.status == TaskStatus.FAILED
        ]


# Sample playbook templates
PLAYBOOK_TEMPLATES = {
    PlaybookType.DROPSHIPPING: {
        "name": "Dropshipping Store Setup",
        "description": "Complete setup for a dropshipping business",
        "tasks": [
            {
                "id": "market_research",
                "name": "Market Research",
                "agent": "research",
                "action": "analyze_market",
                "parameters": {"depth": "comprehensive"},
            },
            {
                "id": "find_products",
                "name": "Find Winning Products",
                "agent": "supplier",
                "action": "search",
                "parameters": {"min_margin": 40, "limit": 20},
                "dependencies": ["market_research"],
            },
            {
                "id": "setup_store",
                "name": "Setup Shopify Store",
                "agent": "commerce",
                "action": "create_store",
                "dependencies": ["find_products"],
            },
            {
                "id": "import_products",
                "name": "Import Products",
                "agent": "supplier",
                "action": "import",
                "dependencies": ["setup_store", "find_products"],
            },
            {
                "id": "create_content",
                "name": "Create Marketing Content",
                "agent": "content",
                "action": "generate_batch",
                "parameters": {"types": ["product_descriptions", "social_posts"]},
                "dependencies": ["import_products"],
            },
            {
                "id": "setup_analytics",
                "name": "Setup Analytics",
                "agent": "analytics",
                "action": "configure",
                "dependencies": ["setup_store"],
            },
        ],
    },
    PlaybookType.SAAS: {
        "name": "SaaS Launch Playbook",
        "description": "Launch a SaaS product",
        "tasks": [
            {
                "id": "validate_idea",
                "name": "Validate Product Idea",
                "agent": "research",
                "action": "validate_market",
            },
            {
                "id": "setup_stripe",
                "name": "Setup Payment Processing",
                "agent": "finance",
                "action": "configure_stripe",
                "dependencies": ["validate_idea"],
            },
            {
                "id": "create_landing",
                "name": "Create Landing Page",
                "agent": "content",
                "action": "generate_landing_page",
                "dependencies": ["validate_idea"],
            },
            {
                "id": "setup_legal",
                "name": "Generate Legal Documents",
                "agent": "legal",
                "action": "generate_suite",
                "parameters": {"documents": ["privacy_policy", "terms_of_service"]},
                "dependencies": ["create_landing"],
            },
            {
                "id": "launch_marketing",
                "name": "Launch Marketing Campaign",
                "agent": "content",
                "action": "create_campaign",
                "dependencies": ["create_landing", "setup_legal"],
            },
        ],
    },
}
```

---

### Task 20.2: Playbook Loader

**File: `src/business/playbook_loader.py`**

```python
"""
Playbook Loader - Load and validate playbooks from YAML.
"""
import os
from pathlib import Path
from typing import Optional
import yaml
from src.business.playbook_models import (
    PlaybookDefinition, PlaybookType, TaskDefinition, PLAYBOOK_TEMPLATES
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PlaybookLoader:
    """Load playbooks from YAML files or templates."""

    def __init__(self, playbooks_dir: str = None):
        self.playbooks_dir = Path(playbooks_dir) if playbooks_dir else Path("config/playbooks")
        self._cache: dict[str, PlaybookDefinition] = {}

    def load_from_file(self, filepath: str) -> Optional[PlaybookDefinition]:
        """Load a playbook from a YAML file."""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            return self._parse_playbook(data)
        except Exception as e:
            logger.error(f"Error loading playbook from {filepath}: {e}")
            return None

    def load_from_template(self, playbook_type: PlaybookType) -> Optional[PlaybookDefinition]:
        """Load a playbook from built-in templates."""
        template = PLAYBOOK_TEMPLATES.get(playbook_type)
        if not template:
            logger.warning(f"No template for type: {playbook_type}")
            return None
        
        return self._parse_playbook({
            "id": f"template_{playbook_type.value}",
            "type": playbook_type.value,
            **template,
        })

    def load_all(self) -> list[PlaybookDefinition]:
        """Load all playbooks from the playbooks directory."""
        playbooks = []
        
        if self.playbooks_dir.exists():
            for file in self.playbooks_dir.glob("*.yaml"):
                pb = self.load_from_file(str(file))
                if pb:
                    playbooks.append(pb)
                    self._cache[pb.id] = pb
        
        return playbooks

    def get_cached(self, playbook_id: str) -> Optional[PlaybookDefinition]:
        """Get a cached playbook by ID."""
        return self._cache.get(playbook_id)

    def _parse_playbook(self, data: dict) -> PlaybookDefinition:
        """Parse raw data into a PlaybookDefinition."""
        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(TaskDefinition(
                id=task_data["id"],
                name=task_data.get("name", task_data["id"]),
                description=task_data.get("description", ""),
                agent=task_data.get("agent", ""),
                action=task_data.get("action", ""),
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", []),
                conditions=task_data.get("conditions", []),
                timeout_seconds=task_data.get("timeout_seconds", 300),
                retry_count=task_data.get("retry_count", 3),
                on_failure=task_data.get("on_failure", "continue"),
            ))

        try:
            pb_type = PlaybookType(data.get("type", "custom"))
        except ValueError:
            pb_type = PlaybookType.CUSTOM

        return PlaybookDefinition(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unnamed Playbook"),
            playbook_type=pb_type,
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            tasks=tasks,
            triggers=data.get("triggers", []),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )

    def validate_playbook(self, playbook: PlaybookDefinition) -> tuple[bool, list[str]]:
        """Validate a playbook definition."""
        errors = []
        task_ids = {t.id for t in playbook.tasks}

        for task in playbook.tasks:
            # Check dependencies exist
            for dep in task.dependencies:
                if dep not in task_ids:
                    errors.append(f"Task '{task.id}' has unknown dependency: {dep}")

            # Check for circular dependencies
            if self._has_circular_dependency(task.id, playbook):
                errors.append(f"Task '{task.id}' has circular dependency")

            # Check agent is specified
            if not task.agent:
                errors.append(f"Task '{task.id}' has no agent specified")

        return len(errors) == 0, errors

    def _has_circular_dependency(
        self, task_id: str, playbook: PlaybookDefinition, visited: set = None
    ) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()
        
        if task_id in visited:
            return True
        
        visited.add(task_id)
        task = playbook.get_task(task_id)
        
        if task:
            for dep in task.dependencies:
                if self._has_circular_dependency(dep, playbook, visited.copy()):
                    return True
        
        return False
```

---

### Task 20.3: Playbook Executor

**File: `src/business/playbook_executor.py`**

```python
"""
Playbook Executor - Run playbooks and manage task execution.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional, Callable
from src.business.playbook_models import (
    PlaybookDefinition, PlaybookRun, TaskDefinition, TaskExecution,
    TaskStatus, TriggerType
)
from src.utils.logging import get_logger

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
            started_at=datetime.utcnow(),
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

        run.completed_at = datetime.utcnow()

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
        execution.started_at = datetime.utcnow()
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
            execution.completed_at = datetime.utcnow()
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
            execution.completed_at = datetime.utcnow()

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
            execution.completed_at = datetime.utcnow()
            
            # Retry if attempts remaining
            if execution.attempts < task.retry_count:
                execution.status = TaskStatus.PENDING
                return await self._execute_task(task, run)

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()

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
        run.completed_at = datetime.utcnow()
        return True
```

---

### Task 20.4: Playbook API Routes

**File: `src/api/routes/playbook.py`**

```python
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
```

---

### Task 20.5: Tests

**File: `tests/test_playbook.py`**

```python
"""Tests for Playbook System."""
import pytest
from src.business.playbook_models import (
    PlaybookDefinition, PlaybookType, TaskDefinition, TaskStatus
)
from src.business.playbook_loader import PlaybookLoader
from src.business.playbook_executor import PlaybookExecutor


@pytest.fixture
def loader():
    return PlaybookLoader()


@pytest.fixture
def executor():
    return PlaybookExecutor()


@pytest.fixture
def sample_playbook():
    return PlaybookDefinition(
        id="test_pb",
        name="Test Playbook",
        playbook_type=PlaybookType.CUSTOM,
        description="Test",
        tasks=[
            TaskDefinition(
                id="task_1",
                name="First Task",
                description="",
                agent="test_agent",
                action="test_action",
            ),
            TaskDefinition(
                id="task_2",
                name="Second Task",
                description="",
                agent="test_agent",
                action="test_action",
                dependencies=["task_1"],
            ),
        ],
    )


class TestPlaybookLoader:
    def test_load_from_template(self, loader):
        pb = loader.load_from_template(PlaybookType.DROPSHIPPING)
        assert pb is not None
        assert pb.playbook_type == PlaybookType.DROPSHIPPING
        assert len(pb.tasks) > 0

    def test_load_saas_template(self, loader):
        pb = loader.load_from_template(PlaybookType.SAAS)
        assert pb is not None
        assert "SaaS" in pb.name

    def test_validate_playbook(self, loader, sample_playbook):
        valid, errors = loader.validate_playbook(sample_playbook)
        assert valid is True
        assert len(errors) == 0

    def test_validate_invalid_dependency(self, loader):
        pb = PlaybookDefinition(
            id="bad_pb",
            name="Bad",
            playbook_type=PlaybookType.CUSTOM,
            description="",
            tasks=[
                TaskDefinition(
                    id="task_1",
                    name="Task",
                    description="",
                    agent="test",
                    action="test",
                    dependencies=["nonexistent"],
                ),
            ],
        )
        valid, errors = loader.validate_playbook(pb)
        assert valid is False
        assert len(errors) > 0


class MockAgent:
    async def execute(self, action, context):
        return {"success": True, "action": action}


class TestPlaybookExecutor:
    @pytest.mark.asyncio
    async def test_execute_playbook(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        
        assert run.status == TaskStatus.COMPLETED
        assert run.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_task_dependencies(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        
        task_1 = run.task_executions["task_1"]
        task_2 = run.task_executions["task_2"]
        
        assert task_1.status == TaskStatus.COMPLETED
        assert task_2.status == TaskStatus.COMPLETED
        # Task 2 should complete after Task 1
        assert task_2.started_at >= task_1.completed_at

    @pytest.mark.asyncio
    async def test_get_run(self, executor, sample_playbook):
        executor.register_agent("test_agent", MockAgent())
        
        run = await executor.execute(sample_playbook, "biz_1")
        retrieved = await executor.get_run(run.id)
        
        assert retrieved is not None
        assert retrieved.id == run.id

    @pytest.mark.asyncio
    async def test_cancel_run(self, executor):
        # Can't really test cancellation mid-run without async tricks
        result = await executor.cancel_run("nonexistent")
        assert result is False
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Templates load | Dropshipping, SaaS templates available |
| YAML loading | Playbooks load from files |
| Validation works | Invalid playbooks rejected |
| Tasks execute | Tasks run with agents |
| Dependencies respected | Tasks wait for dependencies |
| Progress tracked | Run status and progress available |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/business/playbook_models.py` | Data models and templates |
| `src/business/playbook_loader.py` | Load/validate playbooks |
| `src/business/playbook_executor.py` | Execute playbooks |
| `src/api/routes/playbook.py` | REST API endpoints |
| `tests/test_playbook.py` | Unit tests |
