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
from src.utils.structured_logging import get_logger

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

    # Legacy methods for backward compatibility
    def load_playbook(self, name: str) -> dict | None:
        """Loads a specific playbook by filename (without extension)."""
        file_path = self.playbooks_dir / f"{name}.yaml"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def list_playbooks(self) -> list[str]:
        """Lists all available business blueprints."""
        if not self.playbooks_dir.exists():
            return []
        return [f.stem for f in self.playbooks_dir.glob("*.yaml")]
