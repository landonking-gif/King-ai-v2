"""
Sandboxed executor for skill execution with validation and provenance tracking.
Module: code-exec/service/executor.py
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import anyio
from anyio import fail_after

from .config import settings
from .models import (
    ExecutionLog,
    ExecutionRequest,
    ExecutionResult,
    ProvenanceRecord,
    SafetyFlag,
)
from .registry import SkillRegistry, SkillValidationError

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution errors."""

    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when execution exceeds timeout."""

    pass


class SandboxedExecutor:
    """
    Sandboxed executor for deterministic skill execution.

    Responsibilities:
    - Input/output validation
    - Sandboxed execution with timeout
    - Hash generation for provenance
    - Execution logging
    """

    def __init__(self, registry: SkillRegistry) -> None:
        """
        Initialize the executor.

        Args:
            registry: Skill registry instance
        """
        self.registry = registry
        self.execution_logs: Dict[str, List[ExecutionLog]] = {}

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a skill with full validation and provenance tracking.

        Args:
            request: Execution request

        Returns:
            Execution result with provenance

        Raises:
            ExecutionError: If execution fails
            SkillValidationError: If validation fails
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        # Initialize logs for this execution
        self.execution_logs[execution_id] = []

        try:
            # Get skill metadata
            skill_metadata = self.registry.get_skill(request.skill)

            # Check for policy approval requirement
            if skill_metadata.requires_approval:
                await self._log(
                    execution_id,
                    "WARNING",
                    f"Skill '{request.skill}' requires policy approval",
                )
                if SafetyFlag.SIDE_EFFECT in skill_metadata.safety_flags:
                    if settings.require_policy_approval_for_side_effects:
                        raise ExecutionError(
                            f"Skill '{request.skill}' requires policy approval due to "
                            f"side-effect flags. Approval mechanism not implemented."
                        )

            # Validate inputs
            await self._log(execution_id, "INFO", f"Validating inputs for {request.skill}")
            try:
                self.registry.validate_input(request.skill, request.args)
            except SkillValidationError as e:
                await self._log(execution_id, "ERROR", f"Input validation failed: {e}")
                raise

            # Hash inputs
            inputs_hash = self._hash_data(request.args)
            await self._log(execution_id, "INFO", f"Input hash: {inputs_hash[:16]}...")

            # Execute skill handler
            await self._log(execution_id, "INFO", f"Executing skill: {request.skill}")

            result = await self._execute_with_timeout(
                execution_id, request.skill, request.args
            )

            # Validate outputs
            await self._log(execution_id, "INFO", "Validating outputs")
            try:
                self.registry.validate_output(request.skill, result)
            except SkillValidationError as e:
                await self._log(execution_id, "ERROR", f"Output validation failed: {e}")
                raise

            # Hash outputs
            outputs_hash = self._hash_data(result)
            await self._log(execution_id, "INFO", f"Output hash: {outputs_hash[:16]}...")

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create provenance record
            provenance = ProvenanceRecord(
                execution_id=execution_id,
                skill_name=skill_metadata.name,
                skill_version=skill_metadata.version,
                actor_id=request.actor_id,
                actor_type=request.actor_type,
                inputs_hash=inputs_hash,
                outputs_hash=outputs_hash,
                tool_ids=[],  # Would be populated if skill calls external tools
                execution_time_ms=execution_time_ms,
                success=True,
            )

            await self._log(
                execution_id, "INFO", f"Execution completed in {execution_time_ms:.2f}ms"
            )

            # Build execution result
            execution_result = ExecutionResult(
                execution_id=execution_id,
                skill=request.skill,
                result=result,
                logs=self.execution_logs[execution_id],
                inputs_hash=inputs_hash,
                outputs_hash=outputs_hash,
                provenance=provenance,
                execution_time_ms=execution_time_ms,
                success=True,
            )

            return execution_result

        except Exception as e:
            # Log error
            await self._log(execution_id, "ERROR", f"Execution failed: {str(e)}")

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create error provenance
            provenance = ProvenanceRecord(
                execution_id=execution_id,
                skill_name=request.skill,
                skill_version="unknown",
                actor_id=request.actor_id,
                actor_type=request.actor_type,
                inputs_hash=self._hash_data(request.args),
                outputs_hash="",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e),
            )

            # Return error result
            return ExecutionResult(
                execution_id=execution_id,
                skill=request.skill,
                result=None,
                logs=self.execution_logs.get(execution_id, []),
                inputs_hash=self._hash_data(request.args),
                outputs_hash="",
                provenance=provenance,
                execution_time_ms=execution_time_ms,
                success=False,
            )

        finally:
            # Clean up logs after some time (or persist them)
            # For now, keep them in memory for debugging
            pass

    async def _execute_with_timeout(
        self, execution_id: str, skill_name: str, args: Dict[str, Any]
    ) -> Any:
        """
        Execute skill handler with timeout protection.

        Args:
            execution_id: Execution ID for logging
            skill_name: Name of skill to execute
            args: Skill arguments

        Returns:
            Skill execution result

        Raises:
            ExecutionTimeoutError: If execution exceeds timeout
            ExecutionError: If execution fails
        """
        handler = self.registry.get_handler(skill_name)

        try:
            # Execute with timeout
            with fail_after(settings.max_execution_time):
                # Check if handler is async or sync
                if anyio.get_asynclib() == "asyncio":
                    import asyncio
                    import inspect

                    if inspect.iscoroutinefunction(handler):
                        result = await handler(**args)
                    else:
                        # Run sync function in thread pool
                        result = await asyncio.to_thread(handler, **args)
                else:
                    # For trio, use to_thread.run_sync
                    import inspect

                    if inspect.iscoroutinefunction(handler):
                        result = await handler(**args)
                    else:
                        result = await anyio.to_thread.run_sync(handler, **args)

                return result

        except TimeoutError:
            raise ExecutionTimeoutError(
                f"Execution exceeded timeout of {settings.max_execution_time}s"
            )
        except Exception as e:
            raise ExecutionError(f"Handler execution failed: {str(e)}")

    def _hash_data(self, data: Any) -> str:
        """
        Generate SHA-256 hash of data for provenance tracking.

        Args:
            data: Data to hash (will be JSON serialized)

        Returns:
            Hexadecimal hash string
        """
        # Serialize to JSON with sorted keys for deterministic hashing
        json_str = json.dumps(data, sort_keys=True, default=str)

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))

        return hash_obj.hexdigest()

    async def _log(self, execution_id: str, level: str, message: str) -> None:
        """
        Add log entry for an execution.

        Args:
            execution_id: Execution ID
            level: Log level
            message: Log message
        """
        log_entry = ExecutionLog(level=level, message=message)

        if execution_id not in self.execution_logs:
            self.execution_logs[execution_id] = []

        self.execution_logs[execution_id].append(log_entry)

        # Also log to standard logger
        logger_method = getattr(logger, level.lower(), logger.info)
        logger_method(f"[{execution_id[:8]}] {message}")

    def get_logs(self, execution_id: str) -> List[ExecutionLog]:
        """
        Get logs for a specific execution.

        Args:
            execution_id: Execution ID

        Returns:
            List of execution logs
        """
        return self.execution_logs.get(execution_id, [])

    def clear_logs(self, execution_id: Optional[str] = None) -> None:
        """
        Clear execution logs.

        Args:
            execution_id: If provided, clear logs for specific execution.
                         Otherwise, clear all logs.
        """
        if execution_id:
            self.execution_logs.pop(execution_id, None)
        else:
            self.execution_logs.clear()
