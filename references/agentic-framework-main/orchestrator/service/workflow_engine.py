"""
Workflow Engine for executing YAML manifest-based workflows.

Handles manifest loading, validation, step execution, artifact management,
and provenance tracking.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anyio
import httpx
import yaml
from pydantic import ValidationError

from .config import config
from .models import (
    Artifact,
    ArtifactHandleRequest,
    ArtifactHandleResponse,
    ArtifactType,
    ClaimVerification,
    CodePatch,
    ProvenanceRecord,
    ResearchSnippet,
    SubagentRequest,
    SubagentResponse,
    SynthesisResult,
    WorkflowContext,
    WorkflowManifest,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class WorkflowEngineError(Exception):
    """Base exception for workflow engine errors."""

    pass


class ManifestValidationError(WorkflowEngineError):
    """Raised when manifest validation fails."""

    pass


class StepExecutionError(WorkflowEngineError):
    """Raised when step execution fails."""

    pass


class WorkflowEngine:
    """
    Workflow engine that executes manifest-based workflows.

    Responsibilities:
    - Load and validate YAML manifests
    - Execute workflow steps sequentially
    - Manage artifact creation and validation
    - Track provenance for all operations
    - Handle retries and error recovery
    """

    def __init__(
        self,
        manifest_dir: Optional[Path] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Initialize workflow engine.

        Args:
            manifest_dir: Directory containing workflow manifests
            http_client: HTTP client for service calls (created if not provided)
        """
        self.manifest_dir = manifest_dir or Path("./orchestrator/manifests")
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._manifests_cache: Dict[str, WorkflowManifest] = {}

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self.http_client.aclose()

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA256 hash of data for provenance tracking.

        Args:
            data: Data to hash (will be converted to string)

        Returns:
            Hex digest of SHA256 hash
        """
        content = str(data).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _create_provenance(
        self,
        actor_id: str,
        actor_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_ids: Optional[List[str]] = None,
    ) -> ProvenanceRecord:
        """
        Create provenance record for an operation.

        Args:
            actor_id: ID of the actor performing operation
            actor_type: Type of actor (subagent, user, system)
            inputs: Input data
            outputs: Output data
            tool_ids: List of tools/skills used

        Returns:
            Provenance record
        """
        return ProvenanceRecord(
            actor_id=actor_id,
            actor_type=actor_type,
            inputs_hash=self._compute_hash(inputs),
            outputs_hash=self._compute_hash(outputs),
            tool_ids=tool_ids or [],
            timestamp=datetime.utcnow(),
            metadata={},
        )

    async def load_manifest_from_file(self, manifest_name: str) -> WorkflowManifest:
        """
        Load workflow manifest from file.

        Args:
            manifest_name: Name of manifest file (without .yaml extension)

        Returns:
            Parsed and validated workflow manifest

        Raises:
            ManifestValidationError: If manifest is invalid
            FileNotFoundError: If manifest file doesn't exist
        """
        # Check cache first
        if manifest_name in self._manifests_cache:
            logger.info(f"Loaded manifest '{manifest_name}' from cache")
            return self._manifests_cache[manifest_name]

        # Load from file
        manifest_path = self.manifest_dir / f"{manifest_name}.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = yaml.safe_load(f)

            manifest = WorkflowManifest(**manifest_data)
            self._manifests_cache[manifest_name] = manifest
            logger.info(f"Loaded manifest '{manifest_name}' from {manifest_path}")
            return manifest

        except yaml.YAMLError as e:
            raise ManifestValidationError(f"Invalid YAML in manifest: {e}")
        except ValidationError as e:
            raise ManifestValidationError(f"Invalid manifest structure: {e}")

    async def load_manifest_from_yaml(self, yaml_content: str) -> WorkflowManifest:
        """
        Load workflow manifest from YAML string.

        Args:
            yaml_content: YAML manifest content

        Returns:
            Parsed and validated workflow manifest

        Raises:
            ManifestValidationError: If manifest is invalid
        """
        try:
            manifest_data = yaml.safe_load(yaml_content)
            manifest = WorkflowManifest(**manifest_data)
            logger.info(f"Loaded manifest '{manifest.name}' from YAML string")
            return manifest

        except yaml.YAMLError as e:
            raise ManifestValidationError(f"Invalid YAML: {e}")
        except ValidationError as e:
            raise ManifestValidationError(f"Invalid manifest structure: {e}")

    async def _resolve_step_inputs(
        self, step_inputs_spec: List[Any], context: WorkflowContext
    ) -> Dict[str, Any]:
        """
        Resolve inputs for a workflow step.

        Args:
            step_inputs_spec: Input specifications from workflow step
            context: Current workflow context

        Returns:
            Resolved input data dictionary
        """
        resolved_inputs: Dict[str, Any] = {}

        for input_spec in step_inputs_spec:
            name = input_spec.name
            source = input_spec.source

            if source == "user_input":
                # Get from user-provided input
                if name in context.user_input:
                    resolved_inputs[name] = context.user_input[name]
                elif input_spec.required:
                    raise StepExecutionError(
                        f"Required input '{name}' not found in user_input"
                    )

            elif source.startswith("artifact:"):
                # Get from specific artifact
                artifact_id = source.split(":", 1)[1]
                # In production, fetch from memory service
                resolved_inputs[name] = {"artifact_id": artifact_id}

            elif source == "previous_step":
                # Get from previous step's artifacts
                if context.current_step_index > 0:
                    prev_step = context.manifest.steps[context.current_step_index - 1]
                    prev_artifacts = context.step_artifacts.get(prev_step.id, [])
                    if prev_artifacts:
                        resolved_inputs[name] = {"artifact_ids": prev_artifacts}
                elif input_spec.required:
                    raise StepExecutionError(
                        f"No previous step to get input '{name}' from"
                    )

            elif source == "memory":
                # Query from memory service
                # In production, call memory service API
                resolved_inputs[name] = {"source": "memory"}

            else:
                logger.warning(f"Unknown input source: {source}")

        return resolved_inputs

    async def _execute_subagent_step(
        self, step: Any, inputs: Dict[str, Any], context: WorkflowContext
    ) -> SubagentResponse:
        """
        Execute a single subagent step.

        Args:
            step: Workflow step definition
            inputs: Resolved inputs for the step
            context: Current workflow context

        Returns:
            Subagent response with artifacts

        Raises:
            StepExecutionError: If step execution fails
        """
        request = SubagentRequest(
            workflow_id=context.workflow_id,
            step_id=step.id,
            role=step.role,
            capabilities=step.capabilities,
            inputs=inputs,
            timeout=step.timeout,
        )

        logger.info(
            f"Executing subagent step '{step.id}' with role '{step.role}' "
            f"for workflow '{context.workflow_id}'"
        )

        try:
            # Call subagent manager service
            response = await self.http_client.post(
                f"{config.subagent_manager_url}/subagent/execute",
                json=request.model_dump(),
                timeout=step.timeout + 10.0,  # Add buffer to step timeout
            )
            response.raise_for_status()

            result = SubagentResponse(**response.json())
            logger.info(
                f"Step '{step.id}' completed with status '{result.status}', "
                f"produced {len(result.artifacts)} artifacts"
            )
            return result

        except httpx.TimeoutException:
            raise StepExecutionError(f"Step '{step.id}' timed out after {step.timeout}s")
        except httpx.HTTPStatusError as e:
            raise StepExecutionError(
                f"Step '{step.id}' failed with HTTP {e.response.status_code}: "
                f"{e.response.text}"
            )
        except Exception as e:
            raise StepExecutionError(f"Step '{step.id}' failed: {str(e)}")

    async def _validate_and_persist_artifact(
        self,
        artifact_data: Dict[str, Any],
        artifact_type: ArtifactType,
        workflow_id: str,
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate and persist an artifact.

        Args:
            artifact_data: Raw artifact data
            artifact_type: Expected artifact type
            workflow_id: Associated workflow ID

        Returns:
            Tuple of (is_valid, artifact_id, validation_errors)
        """
        request = ArtifactHandleRequest(
            artifact_data=artifact_data,
            artifact_type=artifact_type,
            validate_only=False,
            workflow_id=workflow_id,
        )

        try:
            # Validate artifact structure using Pydantic
            if artifact_type == ArtifactType.RESEARCH_SNIPPET:
                artifact = ResearchSnippet(**artifact_data)
            elif artifact_type == ArtifactType.CLAIM_VERIFICATION:
                artifact = ClaimVerification(**artifact_data)
            elif artifact_type == ArtifactType.CODE_PATCH:
                artifact = CodePatch(**artifact_data)
            elif artifact_type == ArtifactType.SYNTHESIS_RESULT:
                artifact = SynthesisResult(**artifact_data)
            else:
                return False, "", [f"Unknown artifact type: {artifact_type}"]

            # In production, persist to memory service
            # For now, just return validation success
            artifact_id = artifact.id
            logger.info(f"Validated and would persist artifact '{artifact_id}'")
            return True, artifact_id, []

        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            logger.warning(f"Artifact validation failed: {errors}")
            return False, "", errors

    async def execute_workflow(
        self, manifest: WorkflowManifest, user_input: Dict[str, Any]
    ) -> WorkflowContext:
        """
        Execute a complete workflow.

        Args:
            manifest: Workflow manifest to execute
            user_input: User-provided input data

        Returns:
            Final workflow context with results

        Raises:
            WorkflowEngineError: If workflow execution fails
        """
        # Create workflow context
        from uuid import uuid4

        context = WorkflowContext(
            workflow_id=str(uuid4()),
            manifest=manifest,
            status=WorkflowStatus.RUNNING,
            user_input=user_input,
            started_at=datetime.utcnow(),
        )

        logger.info(
            f"Starting workflow '{manifest.name}' (ID: {context.workflow_id}) "
            f"with {len(manifest.steps)} steps"
        )

        try:
            # Execute steps sequentially
            for step_index, step in enumerate(manifest.steps):
                context.current_step_index = step_index
                logger.info(
                    f"Executing step {step_index + 1}/{len(manifest.steps)}: "
                    f"'{step.id}' (role: {step.role})"
                )

                # Resolve inputs for this step
                inputs = await self._resolve_step_inputs(step.inputs, context)

                # Execute step with retries
                last_error: Optional[Exception] = None
                for attempt in range(step.max_retries + 1 if step.retry_on_failure else 1):
                    try:
                        result = await self._execute_subagent_step(step, inputs, context)

                        if result.status == "success":
                            # Store artifacts from this step
                            context.step_artifacts[step.id] = result.artifacts
                            logger.info(
                                f"Step '{step.id}' succeeded, "
                                f"produced artifacts: {result.artifacts}"
                            )
                            break
                        else:
                            last_error = StepExecutionError(
                                result.error_message or "Unknown error"
                            )

                    except Exception as e:
                        last_error = e
                        if attempt < step.max_retries:
                            logger.warning(
                                f"Step '{step.id}' attempt {attempt + 1} failed: {e}. "
                                f"Retrying..."
                            )
                            await anyio.sleep(2.0 ** attempt)  # Exponential backoff
                        else:
                            logger.error(f"Step '{step.id}' failed after all retries")
                            raise

            # Workflow completed successfully
            context.status = WorkflowStatus.COMPLETED
            context.completed_at = datetime.utcnow()
            logger.info(
                f"Workflow '{manifest.name}' (ID: {context.workflow_id}) "
                f"completed successfully"
            )

        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.completed_at = datetime.utcnow()
            context.error_messages.append(str(e))
            logger.error(
                f"Workflow '{manifest.name}' (ID: {context.workflow_id}) failed: {e}"
            )
            raise

        return context

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowContext]:
        """
        Get current status of a workflow execution.

        Args:
            workflow_id: Workflow execution ID

        Returns:
            Current workflow context or None if not found
        """
        # In production, retrieve from database/cache
        # For now, return None (would need state persistence)
        logger.info(f"Getting status for workflow '{workflow_id}'")
        return None

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: Workflow execution ID

        Returns:
            True if cancelled, False if not found or already completed
        """
        # In production, mark workflow as cancelled in database
        logger.info(f"Cancelling workflow '{workflow_id}'")
        return False
