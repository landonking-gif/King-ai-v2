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
