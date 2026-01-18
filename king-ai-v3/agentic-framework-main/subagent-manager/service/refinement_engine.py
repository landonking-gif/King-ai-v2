"""
Refinement Engine for Reflective Agent.

Module: subagent-manager/service/refinement_engine.py

Analyzes failures and determines optimal refinement strategies.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .reflective_agent import (
    ExecutionPlan,
    RefinementAction,
    ReflectionState,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
    """Categories of execution failures."""

    TOOL_ERROR = "tool_error"  # Tool/skill execution failed
    VALIDATION_ERROR = "validation_error"  # Output didn't meet criteria
    TIMEOUT = "timeout"  # Execution timed out
    MISSING_INPUT = "missing_input"  # Required input was missing
    PERMISSION_ERROR = "permission_error"  # Lacked required permissions
    RESOURCE_ERROR = "resource_error"  # Resource unavailable
    LOGIC_ERROR = "logic_error"  # Approach was fundamentally flawed
    PARTIAL_SUCCESS = "partial_success"  # Some steps succeeded
    UNKNOWN = "unknown"  # Couldn't determine cause


class RefinementStrategy(str, Enum):
    """Available refinement strategies."""

    RETRY_SAME = "retry_same"  # Retry exact same approach
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with delay
    MODIFY_INPUTS = "modify_inputs"  # Change input parameters
    MODIFY_APPROACH = "modify_approach"  # Change overall strategy
    SKIP_FAILED_STEP = "skip_failed_step"  # Skip problematic step
    DECOMPOSE_TASK = "decompose_task"  # Break into smaller tasks
    USE_FALLBACK = "use_fallback"  # Try fallback approach
    ESCALATE = "escalate"  # Ask for human help
    ABORT = "abort"  # Give up


@dataclass
class FailureAnalysis:
    """Analysis of a failure."""

    category: FailureCategory
    severity: float  # 0.0 (minor) to 1.0 (critical)
    is_transient: bool  # Could succeed on retry
    root_cause: str
    affected_steps: List[str]
    suggested_strategies: List[RefinementStrategy]


class RefinementEngine:
    """
    Engine for analyzing failures and determining refinement strategies.

    Uses pattern matching and heuristics to choose optimal refinement
    approach based on failure characteristics.
    """

    # Error patterns mapped to failure categories
    ERROR_PATTERNS = {
        FailureCategory.TOOL_ERROR: [
            "tool failed",
            "skill error",
            "execution failed",
            "command not found",
            "subprocess error",
        ],
        FailureCategory.TIMEOUT: [
            "timeout",
            "timed out",
            "took too long",
            "deadline exceeded",
        ],
        FailureCategory.MISSING_INPUT: [
            "missing",
            "not found",
            "required",
            "undefined",
            "null",
        ],
        FailureCategory.PERMISSION_ERROR: [
            "permission denied",
            "access denied",
            "unauthorized",
            "forbidden",
        ],
        FailureCategory.RESOURCE_ERROR: [
            "resource",
            "unavailable",
            "connection",
            "network",
            "rate limit",
        ],
    }

    # Strategy recommendations by failure category
    STRATEGY_MAP = {
        FailureCategory.TOOL_ERROR: [
            RefinementStrategy.USE_FALLBACK,
            RefinementStrategy.MODIFY_APPROACH,
            RefinementStrategy.SKIP_FAILED_STEP,
        ],
        FailureCategory.VALIDATION_ERROR: [
            RefinementStrategy.MODIFY_APPROACH,
            RefinementStrategy.MODIFY_INPUTS,
            RefinementStrategy.DECOMPOSE_TASK,
        ],
        FailureCategory.TIMEOUT: [
            RefinementStrategy.RETRY_WITH_BACKOFF,
            RefinementStrategy.DECOMPOSE_TASK,
            RefinementStrategy.MODIFY_APPROACH,
        ],
        FailureCategory.MISSING_INPUT: [
            RefinementStrategy.MODIFY_INPUTS,
            RefinementStrategy.MODIFY_APPROACH,
        ],
        FailureCategory.PERMISSION_ERROR: [
            RefinementStrategy.ESCALATE,
            RefinementStrategy.USE_FALLBACK,
        ],
        FailureCategory.RESOURCE_ERROR: [
            RefinementStrategy.RETRY_WITH_BACKOFF,
            RefinementStrategy.USE_FALLBACK,
        ],
        FailureCategory.LOGIC_ERROR: [
            RefinementStrategy.MODIFY_APPROACH,
            RefinementStrategy.DECOMPOSE_TASK,
        ],
        FailureCategory.PARTIAL_SUCCESS: [
            RefinementStrategy.SKIP_FAILED_STEP,
            RefinementStrategy.MODIFY_APPROACH,
        ],
        FailureCategory.UNKNOWN: [
            RefinementStrategy.RETRY_SAME,
            RefinementStrategy.MODIFY_APPROACH,
        ],
    }

    def __init__(
        self,
        max_retries_per_strategy: int = 2,
        transient_retry_threshold: float = 0.3,
    ):
        """
        Initialize refinement engine.

        Args:
            max_retries_per_strategy: Max times to try same strategy
            transient_retry_threshold: Severity below which to assume transient
        """
        self.max_retries_per_strategy = max_retries_per_strategy
        self.transient_retry_threshold = transient_retry_threshold
        self._strategy_attempts: Dict[RefinementStrategy, int] = {}

    def analyze_failure(
        self,
        state: ReflectionState,
        execution_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[ValidationResult] = None,
    ) -> FailureAnalysis:
        """
        Analyze a failure to determine its characteristics.

        Args:
            state: Current reflection state
            execution_result: Results from execution
            validation_result: Validation outcome

        Returns:
            FailureAnalysis with category and recommendations
        """
        # Collect all error information
        errors: List[str] = []

        if execution_result:
            errors.extend(execution_result.get("errors", []))

        if validation_result:
            errors.extend(validation_result.errors)

        # Determine failure category
        category = self._categorize_failure(errors, execution_result, validation_result)

        # Calculate severity
        severity = self._calculate_severity(
            category, execution_result, validation_result
        )

        # Check if transient
        is_transient = self._is_transient(category, severity, errors)

        # Determine root cause
        root_cause = self._determine_root_cause(
            category, errors, execution_result, validation_result
        )

        # Find affected steps
        affected_steps = self._find_affected_steps(execution_result, state.current_plan)

        # Get suggested strategies
        suggested_strategies = self._get_suggested_strategies(
            category, severity, is_transient, state
        )

        return FailureAnalysis(
            category=category,
            severity=severity,
            is_transient=is_transient,
            root_cause=root_cause,
            affected_steps=affected_steps,
            suggested_strategies=suggested_strategies,
        )

    def choose_refinement_action(
        self,
        analysis: FailureAnalysis,
        state: ReflectionState,
        strategy_preference: str = "adaptive",
    ) -> RefinementAction:
        """
        Choose the best refinement action based on analysis.

        Args:
            analysis: Failure analysis
            state: Current reflection state
            strategy_preference: adaptive/conservative/aggressive

        Returns:
            RefinementAction to take
        """
        # Check iteration budget
        iterations_remaining = state.max_iterations - state.iteration

        if iterations_remaining <= 0:
            return RefinementAction(
                action_type="abort",
                reasoning="No iterations remaining",
            )

        # Get strategy based on preference
        strategy = self._select_strategy(
            analysis, iterations_remaining, strategy_preference
        )

        # Convert strategy to action
        return self._strategy_to_action(strategy, analysis, state)

    def _categorize_failure(
        self,
        errors: List[str],
        execution_result: Optional[Dict[str, Any]],
        validation_result: Optional[ValidationResult],
    ) -> FailureCategory:
        """Categorize the type of failure."""
        error_text = " ".join(errors).lower()

        # Check error patterns
        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_text:
                    return category

        # Check validation-specific failures
        if validation_result and not validation_result.is_valid:
            if validation_result.score > 0.5:
                return FailureCategory.PARTIAL_SUCCESS
            return FailureCategory.VALIDATION_ERROR

        # Check execution results
        if execution_result:
            success_rate = execution_result.get("success_rate", 0)
            if success_rate > 0 and success_rate < 1:
                return FailureCategory.PARTIAL_SUCCESS
            if execution_result.get("steps_failed"):
                return FailureCategory.TOOL_ERROR

        return FailureCategory.UNKNOWN

    def _calculate_severity(
        self,
        category: FailureCategory,
        execution_result: Optional[Dict[str, Any]],
        validation_result: Optional[ValidationResult],
    ) -> float:
        """Calculate failure severity (0.0 to 1.0)."""
        # Base severity by category
        base_severity = {
            FailureCategory.TOOL_ERROR: 0.6,
            FailureCategory.VALIDATION_ERROR: 0.5,
            FailureCategory.TIMEOUT: 0.4,
            FailureCategory.MISSING_INPUT: 0.5,
            FailureCategory.PERMISSION_ERROR: 0.8,
            FailureCategory.RESOURCE_ERROR: 0.4,
            FailureCategory.LOGIC_ERROR: 0.9,
            FailureCategory.PARTIAL_SUCCESS: 0.3,
            FailureCategory.UNKNOWN: 0.5,
        }.get(category, 0.5)

        # Adjust based on validation score
        if validation_result:
            # Lower score = higher severity
            score_factor = 1.0 - validation_result.score
            base_severity = (base_severity + score_factor) / 2

        # Adjust based on success rate
        if execution_result:
            success_rate = execution_result.get("success_rate", 0)
            rate_factor = 1.0 - success_rate
            base_severity = (base_severity + rate_factor) / 2

        return min(1.0, max(0.0, base_severity))

    def _is_transient(
        self,
        category: FailureCategory,
        severity: float,
        errors: List[str],
    ) -> bool:
        """Determine if failure is likely transient."""
        # Some categories are typically transient
        transient_categories = {
            FailureCategory.TIMEOUT,
            FailureCategory.RESOURCE_ERROR,
        }

        if category in transient_categories:
            return True

        # Low severity failures might be transient
        if severity < self.transient_retry_threshold:
            return True

        # Check for transient keywords
        transient_keywords = ["temporary", "retry", "again", "intermittent"]
        error_text = " ".join(errors).lower()

        return any(kw in error_text for kw in transient_keywords)

    def _determine_root_cause(
        self,
        category: FailureCategory,
        errors: List[str],
        execution_result: Optional[Dict[str, Any]],
        validation_result: Optional[ValidationResult],
    ) -> str:
        """Determine the root cause of the failure."""
        if errors:
            return errors[0]

        if validation_result and validation_result.errors:
            return validation_result.errors[0]

        if execution_result and execution_result.get("steps_failed"):
            failed_steps = execution_result["steps_failed"]
            return f"Steps failed: {', '.join(failed_steps)}"

        return f"Failure category: {category.value}"

    def _find_affected_steps(
        self,
        execution_result: Optional[Dict[str, Any]],
        plan: Optional[ExecutionPlan],
    ) -> List[str]:
        """Find which steps were affected by the failure."""
        if not execution_result:
            return []

        failed_steps = execution_result.get("steps_failed", [])

        if failed_steps:
            return failed_steps

        # If no explicit failures, check outputs
        if plan and plan.steps:
            outputs = execution_result.get("outputs", {})
            missing = [
                step.step_id
                for step in plan.steps
                if step.step_id not in outputs
            ]
            return missing

        return []

    def _get_suggested_strategies(
        self,
        category: FailureCategory,
        severity: float,
        is_transient: bool,
        state: ReflectionState,
    ) -> List[RefinementStrategy]:
        """Get suggested refinement strategies."""
        strategies = list(self.STRATEGY_MAP.get(category, []))

        # If transient, prioritize retry
        if is_transient:
            strategies.insert(0, RefinementStrategy.RETRY_SAME)

        # If high severity, consider escalation/abort
        if severity > 0.8:
            if RefinementStrategy.ESCALATE not in strategies:
                strategies.append(RefinementStrategy.ESCALATE)
            if RefinementStrategy.ABORT not in strategies:
                strategies.append(RefinementStrategy.ABORT)

        # Filter out already exhausted strategies
        strategies = [
            s for s in strategies
            if self._strategy_attempts.get(s, 0) < self.max_retries_per_strategy
        ]

        return strategies if strategies else [RefinementStrategy.ABORT]

    def _select_strategy(
        self,
        analysis: FailureAnalysis,
        iterations_remaining: int,
        preference: str,
    ) -> RefinementStrategy:
        """Select best strategy based on analysis and preference."""
        strategies = analysis.suggested_strategies

        if not strategies:
            return RefinementStrategy.ABORT

        # For conservative: prefer retries and minimal changes
        if preference == "conservative":
            conservative_order = [
                RefinementStrategy.RETRY_SAME,
                RefinementStrategy.RETRY_WITH_BACKOFF,
                RefinementStrategy.MODIFY_INPUTS,
                RefinementStrategy.SKIP_FAILED_STEP,
            ]
            for s in conservative_order:
                if s in strategies:
                    return s

        # For aggressive: prefer significant changes
        elif preference == "aggressive":
            aggressive_order = [
                RefinementStrategy.MODIFY_APPROACH,
                RefinementStrategy.DECOMPOSE_TASK,
                RefinementStrategy.USE_FALLBACK,
            ]
            for s in aggressive_order:
                if s in strategies:
                    return s

        # For adaptive: consider iterations remaining
        else:
            if iterations_remaining > 2:
                # More room to experiment
                if analysis.is_transient:
                    return strategies[0]  # Try first suggestion
                if RefinementStrategy.MODIFY_APPROACH in strategies:
                    return RefinementStrategy.MODIFY_APPROACH
            elif iterations_remaining == 1:
                # Last chance - try something different
                if RefinementStrategy.USE_FALLBACK in strategies:
                    return RefinementStrategy.USE_FALLBACK
                if RefinementStrategy.MODIFY_APPROACH in strategies:
                    return RefinementStrategy.MODIFY_APPROACH

        return strategies[0]

    def _strategy_to_action(
        self,
        strategy: RefinementStrategy,
        analysis: FailureAnalysis,
        state: ReflectionState,
    ) -> RefinementAction:
        """Convert strategy to concrete action."""
        # Track strategy usage
        self._strategy_attempts[strategy] = (
            self._strategy_attempts.get(strategy, 0) + 1
        )

        action_map = {
            RefinementStrategy.RETRY_SAME: ("retry", "Retrying same approach"),
            RefinementStrategy.RETRY_WITH_BACKOFF: (
                "retry",
                "Retrying with backoff delay",
            ),
            RefinementStrategy.MODIFY_INPUTS: (
                "modify",
                "Modifying input parameters",
            ),
            RefinementStrategy.MODIFY_APPROACH: (
                "modify",
                "Changing overall approach",
            ),
            RefinementStrategy.SKIP_FAILED_STEP: (
                "modify",
                "Skipping failed steps",
            ),
            RefinementStrategy.DECOMPOSE_TASK: (
                "modify",
                "Breaking task into smaller pieces",
            ),
            RefinementStrategy.USE_FALLBACK: (
                "modify",
                "Using fallback approach",
            ),
            RefinementStrategy.ESCALATE: (
                "escalate",
                "Requesting human assistance",
            ),
            RefinementStrategy.ABORT: ("abort", "Aborting - cannot proceed"),
        }

        action_type, base_reasoning = action_map.get(
            strategy, ("abort", "Unknown strategy")
        )

        reasoning = (
            f"{base_reasoning}. "
            f"Root cause: {analysis.root_cause}. "
            f"Severity: {analysis.severity:.2f}. "
            f"Category: {analysis.category.value}"
        )

        modifications = {}
        new_approach = None

        if strategy == RefinementStrategy.SKIP_FAILED_STEP:
            modifications["skip_steps"] = analysis.affected_steps

        elif strategy == RefinementStrategy.MODIFY_APPROACH:
            new_approach = (
                f"Try alternative approach to avoid {analysis.category.value}. "
                f"Previous failures: {', '.join(analysis.affected_steps)}"
            )

        elif strategy == RefinementStrategy.DECOMPOSE_TASK:
            new_approach = "Break the task into smaller, more manageable subtasks"

        elif strategy == RefinementStrategy.USE_FALLBACK:
            new_approach = "Use fallback methods for failed steps"

        return RefinementAction(
            action_type=action_type,
            reasoning=reasoning,
            modifications=modifications,
            new_approach=new_approach,
        )

    def reset(self) -> None:
        """Reset strategy attempt counters."""
        self._strategy_attempts.clear()
