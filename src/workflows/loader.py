"""
Workflow Loader.

Loads workflow manifests from YAML files.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from src.workflows.models import (
    WorkflowManifest,
    WorkflowStep,
    WorkflowVariable,
    StepDependency,
    ApprovalGate,
    RiskLevel,
    ApprovalType,
)
from src.utils.structured_logging import get_logger

logger = get_logger("workflow_loader")


class WorkflowLoader:
    """
    Loads and caches workflow manifests from YAML files.
    
    Supports:
    - Loading from files
    - Loading from directory
    - Validation
    - Caching
    """
    
    def __init__(self, workflows_path: Optional[Path] = None):
        """
        Initialize workflow loader.
        
        Args:
            workflows_path: Base path for workflow files
        """
        self.workflows_path = workflows_path or Path("config/workflows")
        self.workflows_path.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded workflows
        self._cache: Dict[str, WorkflowManifest] = {}
    
    def load_file(self, file_path: Path) -> WorkflowManifest:
        """
        Load a workflow from a YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed workflow manifest
        """
        with open(file_path) as f:
            data = yaml.safe_load(f)
        
        return self._parse_manifest(data)
    
    def load_by_id(self, workflow_id: str) -> Optional[WorkflowManifest]:
        """
        Load a workflow by ID.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow manifest or None
        """
        # Check cache
        if workflow_id in self._cache:
            return self._cache[workflow_id]
        
        # Search for file
        for file_path in self.workflows_path.rglob("*.yaml"):
            try:
                workflow = self.load_file(file_path)
                if workflow.id == workflow_id:
                    self._cache[workflow_id] = workflow
                    return workflow
            except Exception:
                continue
        
        for file_path in self.workflows_path.rglob("*.yml"):
            try:
                workflow = self.load_file(file_path)
                if workflow.id == workflow_id:
                    self._cache[workflow_id] = workflow
                    return workflow
            except Exception:
                continue
        
        return None
    
    def load_all(self) -> List[WorkflowManifest]:
        """
        Load all workflows from the workflows directory.
        
        Returns:
            List of workflow manifests
        """
        workflows = []
        
        for file_path in self.workflows_path.rglob("*.yaml"):
            try:
                workflow = self.load_file(file_path)
                workflows.append(workflow)
                self._cache[workflow.id] = workflow
            except Exception as e:
                logger.warning(f"Failed to load workflow {file_path}: {e}")
        
        for file_path in self.workflows_path.rglob("*.yml"):
            try:
                workflow = self.load_file(file_path)
                if workflow.id not in self._cache:
                    workflows.append(workflow)
                    self._cache[workflow.id] = workflow
            except Exception as e:
                logger.warning(f"Failed to load workflow {file_path}: {e}")
        
        return workflows
    
    def load_by_category(self, category: str) -> List[WorkflowManifest]:
        """
        Load workflows by category.
        
        Args:
            category: Category name (saas, ecommerce, etc.)
            
        Returns:
            List of matching workflows
        """
        all_workflows = self.load_all()
        return [w for w in all_workflows if w.category == category]
    
    def save_workflow(
        self,
        workflow: WorkflowManifest,
        file_name: Optional[str] = None
    ) -> Path:
        """
        Save a workflow to a YAML file.
        
        Args:
            workflow: Workflow to save
            file_name: Optional file name (defaults to workflow ID)
            
        Returns:
            Path to saved file
        """
        file_name = file_name or f"{workflow.id}.yaml"
        file_path = self.workflows_path / file_name
        
        with open(file_path, 'w') as f:
            yaml.dump(workflow.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        self._cache[workflow.id] = workflow
        
        logger.info(f"Saved workflow to {file_path}")
        return file_path
    
    def validate_file(self, file_path: Path) -> List[str]:
        """
        Validate a workflow file.
        
        Returns:
            List of validation errors
        """
        try:
            workflow = self.load_file(file_path)
            return workflow.validate_workflow()
        except Exception as e:
            return [f"Failed to parse: {e}"]
    
    def clear_cache(self) -> None:
        """Clear the workflow cache."""
        self._cache.clear()
    
    def _parse_manifest(self, data: Dict) -> WorkflowManifest:
        """Parse raw YAML data into a WorkflowManifest."""
        # Parse variables
        variables = []
        for var_data in data.get("variables", []):
            variables.append(WorkflowVariable(**var_data))
        
        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            step = self._parse_step(step_data)
            steps.append(step)
        
        return WorkflowManifest(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unnamed Workflow"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            variables=variables,
            steps=steps,
            parallel_execution=data.get("parallel_execution", True),
            max_parallel_steps=data.get("max_parallel_steps", 5),
            fail_fast=data.get("fail_fast", False),
            final_outputs=data.get("final_outputs", []),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            category=data.get("category", ""),
        )
    
    def _parse_step(self, data: Dict) -> WorkflowStep:
        """Parse step data."""
        # Parse dependencies
        depends_on = []
        for dep in data.get("depends_on", []):
            if isinstance(dep, str):
                depends_on.append(dep)
            else:
                depends_on.append(StepDependency(**dep))
        
        # Parse approval gate
        approval = None
        if data.get("approval"):
            approval_data = data["approval"]
            if isinstance(approval_data.get("type"), str):
                approval_data["type"] = ApprovalType(approval_data["type"])
            approval = ApprovalGate(**approval_data)
        
        # Parse risk level
        risk = RiskLevel.LOW
        if data.get("risk"):
            risk = RiskLevel(data["risk"])
        
        return WorkflowStep(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unnamed Step"),
            description=data.get("description", ""),
            agent=data.get("agent", "research"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", []),
            output_artifact_type=data.get("output_artifact_type", "generic"),
            depends_on=depends_on,
            require_approval=data.get("require_approval", False),
            approval=approval,
            risk=risk,
            condition=data.get("condition"),
            skip_on_failure=data.get("skip_on_failure", False),
            timeout_seconds=data.get("timeout_seconds", 300),
            max_retries=data.get("max_retries", 2),
            tags=data.get("tags", []),
        )


# Global loader instance
_workflow_loader: Optional[WorkflowLoader] = None


def get_workflow_loader() -> WorkflowLoader:
    """Get or create the global workflow loader."""
    global _workflow_loader
    if _workflow_loader is None:
        _workflow_loader = WorkflowLoader()
    return _workflow_loader
