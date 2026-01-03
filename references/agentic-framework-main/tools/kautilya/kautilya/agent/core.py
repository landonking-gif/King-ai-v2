"""
Agent Core for Kautilya.

The main orchestrator that brings together all agentic capabilities:
- Processes user requests using ReAct loop
- Resolves file references intelligently
- Discovers and uses appropriate capabilities
- Handles errors with recovery strategies
- Maintains session memory for learning
- Validates outputs before returning
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .capability_registry import Capability, CapabilityRegistry
from .error_recovery import ErrorAnalysis, ErrorRecoveryEngine, RecoveryStrategy
from .file_resolver import FileMatch, FileResolver
from .output_validator import OutputValidator, ValidationLevel, ValidationResult
from .react_loop import LoopResult, LoopStatus, ReActLoop
from .session_memory import MemoryType, SessionMemory
from .task_planner import ExecutionPlan, TaskPlanner

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a user request."""

    success: bool
    response: str
    confidence: float = 0.0
    files_resolved: List[FileMatch] = field(default_factory=list)
    capabilities_used: List[str] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    react_trace: Optional[LoopResult] = None
    execution_plan: Optional[ExecutionPlan] = None
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "response": self.response,
            "confidence": self.confidence,
            "files_resolved": [str(f.path) for f in self.files_resolved],
            "capabilities_used": self.capabilities_used,
            "duration": self.duration,
            "error": self.error,
        }


class AgentCore:
    """
    Core agent orchestrator for intelligent request processing.

    Brings together all agentic components:
    - FileResolver: Intelligent file path resolution
    - CapabilityRegistry: Dynamic skill/tool discovery
    - TaskPlanner: Complex task decomposition
    - ReActLoop: Iterative reasoning and acting
    - ErrorRecoveryEngine: Error handling and recovery
    - SessionMemory: Learning from interactions
    - OutputValidator: Self-critique before responding

    Usage:
        agent = AgentCore(llm_client)
        result = await agent.process("Extract ESG metrics from @reports/sample.pdf")
    """

    def __init__(
        self,
        llm_client: Any,
        tool_executor: Optional[Any] = None,
        skills_dir: Optional[Path] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize agent core.

        Args:
            llm_client: LLM client for reasoning
            tool_executor: Tool executor for running tools
            skills_dir: Directory containing skills
            max_iterations: Max ReAct iterations
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.verbose = verbose

        # Initialize components
        self.file_resolver = FileResolver()
        self.capability_registry = CapabilityRegistry(skills_dir)
        self.task_planner = TaskPlanner(llm_client, self.capability_registry)
        self.error_recovery = ErrorRecoveryEngine()
        self.session_memory = SessionMemory()
        self.output_validator = OutputValidator(llm_client)
        self.react_loop = ReActLoop(
            llm_client=llm_client,
            capability_registry=self.capability_registry,
            error_recovery=self.error_recovery,
            session_memory=self.session_memory,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        # Discover capabilities at startup
        self._discover_capabilities()

    def _discover_capabilities(self) -> None:
        """Discover all available capabilities."""
        try:
            self.capability_registry.discover_all()
            count = len(self.capability_registry.get_all())
            logger.info(f"Discovered {count} capabilities")
        except Exception as e:
            logger.warning(f"Capability discovery failed: {e}")

    async def process(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        validate_output: bool = True,
    ) -> ProcessingResult:
        """
        Process a user request intelligently.

        Args:
            request: User's request/query
            context: Additional context
            validate_output: Whether to validate output

        Returns:
            ProcessingResult with response and metadata
        """
        start_time = time.time()
        context = context or {}
        task_id = f"task_{int(time.time())}"

        # Start task in session memory
        self.session_memory.start_task(task_id, request)

        try:
            # Step 1: Resolve file references
            files_resolved = self._resolve_files(request)
            if files_resolved:
                # Update request with resolved paths
                request = self._update_request_with_paths(request, files_resolved)
                context["resolved_files"] = [str(f.path) for f in files_resolved]

            # Step 2: Analyze task and create plan
            plan = self.task_planner.create_plan(request, self.capability_registry.get_all())
            analysis = self.task_planner.analyze_task(request)

            # Step 3: Match capabilities
            matched_caps = self.capability_registry.match_capabilities(request)
            if matched_caps:
                context["matched_capabilities"] = [c.name for c in matched_caps]

            # Step 4: Build context from session memory
            memory_context = self.session_memory.build_context_prompt(request)
            if memory_context:
                context["session_context"] = memory_context

            # Step 5: Run ReAct loop
            react_result = await self.react_loop.run(request, context)

            # Step 6: Handle result
            if react_result.status == LoopStatus.COMPLETED:
                response = react_result.final_answer or "Task completed."
                success = True
            elif react_result.status == LoopStatus.WAITING_USER:
                response = react_result.final_answer or "I need more information."
                success = True
            else:
                response = react_result.final_answer or f"Task ended with status: {react_result.status.value}"
                success = False
                if react_result.error:
                    # Try error recovery
                    recovery_response = await self._attempt_recovery(
                        request, react_result.error, context
                    )
                    if recovery_response:
                        response = recovery_response
                        success = True

            # Step 7: Validate output
            validation = None
            if validate_output and success:
                validation = self.output_validator.validate(
                    response, request, level=ValidationLevel.STANDARD
                )
                if not validation.is_valid:
                    # Try to improve response
                    improved = await self._improve_response(
                        request, response, validation
                    )
                    if improved:
                        response = improved

            # Step 8: Record completion
            capabilities_used = [
                step.action_name
                for step in react_result.steps
                if step.action_name
            ]

            self.session_memory.complete_task(task_id, {"response": response})

            return ProcessingResult(
                success=success,
                response=response,
                confidence=validation.confidence if validation else 0.8,
                files_resolved=files_resolved,
                capabilities_used=capabilities_used,
                validation=validation,
                react_trace=react_result,
                execution_plan=plan,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.session_memory.fail_task(task_id, str(e))

            # Attempt recovery
            error_analysis = self.error_recovery.analyze_error(e, request, context)
            error_message = self.error_recovery.format_error_for_user(error_analysis)

            return ProcessingResult(
                success=False,
                response=f"I encountered an issue: {str(e)}",
                duration=time.time() - start_time,
                error=error_message,
            )

    def _resolve_files(self, request: str) -> List[FileMatch]:
        """Resolve file references in the request."""
        resolved = []

        # Extract @file references
        references = self.file_resolver.extract_file_references(request)

        for ref in references:
            # Check cache first
            cached_path = self.session_memory.get_cached_file(ref)
            if cached_path:
                resolved.append(
                    FileMatch(
                        path=Path(cached_path),
                        confidence=1.0,
                        match_type="cached",
                        original_reference=ref,
                    )
                )
                continue

            try:
                match = self.file_resolver.resolve(ref)
                resolved.append(match)
                # Cache the resolution
                self.session_memory.cache_file_resolution(ref, str(match.path))
                logger.info(f"Resolved '{ref}' -> {match.path} ({match.match_type})")
            except FileNotFoundError as e:
                logger.warning(f"Could not resolve file reference: {ref}")
                # Store the error for context
                self.session_memory.add(
                    MemoryType.ERROR,
                    {"type": "file_not_found", "reference": ref, "error": str(e)},
                )

        return resolved

    def _update_request_with_paths(
        self, request: str, resolved_files: List[FileMatch]
    ) -> str:
        """Update request with resolved file paths."""
        updated = request

        for match in resolved_files:
            # Replace @reference with actual path
            ref = match.original_reference
            if ref:
                updated = updated.replace(f"@{ref}", str(match.path))
                updated = updated.replace(f"@\"{ref}\"", str(match.path))

        return updated

    async def _attempt_recovery(
        self,
        request: str,
        error: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Attempt to recover from an error."""
        try:
            # Create error from string
            exc = Exception(error)
            analysis = self.error_recovery.analyze_error(exc, request, context)

            if not analysis.is_recoverable:
                return None

            best_action = analysis.get_best_action()
            if not best_action:
                return None

            if best_action.strategy == RecoveryStrategy.ALTERNATIVE_CAPABILITY:
                # Try alternative capability
                alternatives = best_action.parameters.get("alternatives", [])
                for alt_name in alternatives:
                    cap = self.capability_registry.get(alt_name)
                    if cap:
                        logger.info(f"Trying alternative capability: {alt_name}")
                        # Run with alternative
                        context["use_capability"] = alt_name
                        result = await self.react_loop.run(request, context)
                        if result.status == LoopStatus.COMPLETED:
                            self.error_recovery.record_recovery_attempt(
                                exc, best_action.strategy, True
                            )
                            return result.final_answer

            elif best_action.strategy == RecoveryStrategy.SIMPLIFY_TASK:
                # Decompose and try simpler tasks
                subtasks = self.task_planner.decompose_complex_task(request)
                if len(subtasks) > 1:
                    results = []
                    for subtask in subtasks:
                        sub_result = await self.react_loop.run(subtask, context)
                        if sub_result.final_answer:
                            results.append(sub_result.final_answer)
                    if results:
                        self.error_recovery.record_recovery_attempt(
                            exc, best_action.strategy, True
                        )
                        return "\n\n".join(results)

            elif best_action.strategy == RecoveryStrategy.FALLBACK_RESPONSE:
                # Generate best-effort response using LLM
                fallback_prompt = f"""The following task could not be fully completed due to: {error}

Original task: {request}

Please provide the best possible response with the information available.
Acknowledge any limitations in your response."""

                response = await self.llm_client.chat(fallback_prompt)
                self.error_recovery.record_recovery_attempt(
                    exc, best_action.strategy, True
                )
                return response

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")

        return None

    async def _improve_response(
        self,
        request: str,
        response: str,
        validation: ValidationResult,
    ) -> Optional[str]:
        """Try to improve a response based on validation issues."""
        if not validation.issues:
            return None

        issues_text = "\n".join(str(i) for i in validation.issues[:3])
        suggestions_text = "\n".join(validation.suggestions[:3])

        improvement_prompt = f"""Your previous response had some issues:

{issues_text}

Suggestions:
{suggestions_text}

Original task: {request}
Original response: {response[:500]}

Please provide an improved response that addresses these issues."""

        try:
            improved = await self.llm_client.chat(improvement_prompt)
            return improved
        except Exception as e:
            logger.warning(f"Response improvement failed: {e}")
            return None

    def get_capabilities(self) -> List[Capability]:
        """Get all available capabilities."""
        return self.capability_registry.get_all()

    def get_capability(self, name: str) -> Optional[Capability]:
        """Get a specific capability by name."""
        return self.capability_registry.get(name)

    def add_capability(self, capability: Capability) -> None:
        """Add a new capability."""
        self.capability_registry.register(capability)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return self.session_memory.get_summary()

    def clear_session(self) -> None:
        """Clear session memory."""
        self.session_memory = SessionMemory()

    async def simple_query(self, query: str) -> str:
        """
        Simple query interface for quick responses.

        Args:
            query: User query

        Returns:
            Response string
        """
        result = await self.process(query, validate_output=False)
        return result.response
