"""
Error Recovery Engine for Kautilya.

Provides intelligent error handling and recovery:
- Classifies error types
- Suggests recovery strategies
- Implements automatic retry with different approaches
- Learns from failures to improve future attempts
"""

import logging
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur."""

    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_INPUT = "invalid_input"
    CAPABILITY_NOT_FOUND = "capability_not_found"
    CAPABILITY_FAILED = "capability_failed"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    MEMORY_ERROR = "memory_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Strategies for recovering from errors."""

    RETRY_SAME = "retry_same"  # Retry with same parameters
    RETRY_MODIFIED = "retry_modified"  # Retry with modified parameters
    ALTERNATIVE_CAPABILITY = "alternative_capability"  # Use different capability
    SIMPLIFY_TASK = "simplify_task"  # Break down task further
    REQUEST_USER_INPUT = "request_user_input"  # Ask user for clarification
    SKIP_STEP = "skip_step"  # Skip and continue with next step
    FALLBACK_RESPONSE = "fallback_response"  # Provide best-effort response
    ABORT = "abort"  # Cannot recover, abort task


@dataclass
class RecoveryAction:
    """A specific recovery action to take."""

    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5  # How confident we are this will work
    requires_user_confirmation: bool = False

    def __str__(self) -> str:
        return f"{self.strategy.value}: {self.description}"


@dataclass
class ErrorAnalysis:
    """Analysis of an error and potential recovery options."""

    category: ErrorCategory
    message: str
    original_error: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    is_recoverable: bool = True

    def get_best_action(self) -> Optional[RecoveryAction]:
        """Get the recovery action with highest confidence."""
        if not self.recovery_actions:
            return None
        return max(self.recovery_actions, key=lambda a: a.confidence)


class ErrorRecoveryEngine:
    """
    Intelligent error recovery engine.

    Analyzes errors and suggests/executes recovery strategies:
    - Pattern matching on error messages
    - Context-aware recovery suggestions
    - Automatic retry with exponential backoff
    - Learning from successful recoveries
    """

    # Error patterns and their classifications
    ERROR_PATTERNS = {
        ErrorCategory.FILE_NOT_FOUND: [
            r"file not found",
            r"no such file",
            r"path does not exist",
            r"filenotfounderror",
            r"cannot find",
        ],
        ErrorCategory.PERMISSION_DENIED: [
            r"permission denied",
            r"access denied",
            r"not authorized",
            r"forbidden",
        ],
        ErrorCategory.INVALID_INPUT: [
            r"invalid input",
            r"invalid argument",
            r"validation error",
            r"type error",
            r"value error",
        ],
        ErrorCategory.CAPABILITY_NOT_FOUND: [
            r"skill not found",
            r"tool not found",
            r"capability not found",
            r"no handler",
            r"unknown command",
        ],
        ErrorCategory.TIMEOUT: [
            r"timeout",
            r"timed out",
            r"deadline exceeded",
            r"request took too long",
        ],
        ErrorCategory.RATE_LIMIT: [
            r"rate limit",
            r"too many requests",
            r"429",
            r"quota exceeded",
        ],
        ErrorCategory.API_ERROR: [
            r"api error",
            r"service unavailable",
            r"internal server error",
            r"500",
            r"502",
            r"503",
        ],
        ErrorCategory.NETWORK_ERROR: [
            r"connection error",
            r"network error",
            r"dns",
            r"unreachable",
            r"connection refused",
        ],
        ErrorCategory.PARSING_ERROR: [
            r"parse error",
            r"json",
            r"syntax error",
            r"malformed",
            r"decode error",
        ],
        ErrorCategory.MEMORY_ERROR: [
            r"out of memory",
            r"memory error",
            r"allocation failed",
        ],
    }

    # Recovery strategies by error category
    RECOVERY_STRATEGIES = {
        ErrorCategory.FILE_NOT_FOUND: [
            (RecoveryStrategy.RETRY_MODIFIED, "Search for similar files", 0.8),
            (RecoveryStrategy.REQUEST_USER_INPUT, "Ask user to provide correct path", 0.9),
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Try web search instead", 0.3),
        ],
        ErrorCategory.PERMISSION_DENIED: [
            (RecoveryStrategy.REQUEST_USER_INPUT, "Request elevated permissions", 0.7),
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Use alternative method", 0.5),
            (RecoveryStrategy.ABORT, "Cannot proceed without permissions", 0.3),
        ],
        ErrorCategory.INVALID_INPUT: [
            (RecoveryStrategy.RETRY_MODIFIED, "Modify input format", 0.7),
            (RecoveryStrategy.SIMPLIFY_TASK, "Break down into simpler steps", 0.6),
            (RecoveryStrategy.REQUEST_USER_INPUT, "Clarify input requirements", 0.8),
        ],
        ErrorCategory.CAPABILITY_NOT_FOUND: [
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Find alternative capability", 0.8),
            (RecoveryStrategy.FALLBACK_RESPONSE, "Use LLM for general response", 0.6),
            (RecoveryStrategy.REQUEST_USER_INPUT, "Ask user which capability to use", 0.7),
        ],
        ErrorCategory.CAPABILITY_FAILED: [
            (RecoveryStrategy.RETRY_SAME, "Retry the capability", 0.5),
            (RecoveryStrategy.RETRY_MODIFIED, "Retry with different parameters", 0.6),
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Try alternative capability", 0.7),
        ],
        ErrorCategory.TIMEOUT: [
            (RecoveryStrategy.RETRY_SAME, "Retry with longer timeout", 0.6),
            (RecoveryStrategy.SIMPLIFY_TASK, "Simplify the request", 0.5),
            (RecoveryStrategy.SKIP_STEP, "Skip and continue", 0.4),
        ],
        ErrorCategory.RATE_LIMIT: [
            (RecoveryStrategy.RETRY_SAME, "Wait and retry", 0.9),
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Use alternative provider", 0.7),
        ],
        ErrorCategory.API_ERROR: [
            (RecoveryStrategy.RETRY_SAME, "Retry the request", 0.6),
            (RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Try alternative API", 0.5),
            (RecoveryStrategy.FALLBACK_RESPONSE, "Provide cached/fallback response", 0.4),
        ],
        ErrorCategory.NETWORK_ERROR: [
            (RecoveryStrategy.RETRY_SAME, "Retry connection", 0.7),
            (RecoveryStrategy.FALLBACK_RESPONSE, "Use offline/cached data", 0.5),
        ],
        ErrorCategory.PARSING_ERROR: [
            (RecoveryStrategy.RETRY_MODIFIED, "Try different parser", 0.6),
            (RecoveryStrategy.SIMPLIFY_TASK, "Extract partial data", 0.5),
            (RecoveryStrategy.FALLBACK_RESPONSE, "Return raw content", 0.4),
        ],
        ErrorCategory.UNKNOWN: [
            (RecoveryStrategy.RETRY_SAME, "Retry the operation", 0.4),
            (RecoveryStrategy.REQUEST_USER_INPUT, "Ask user for guidance", 0.6),
            (RecoveryStrategy.FALLBACK_RESPONSE, "Provide best-effort response", 0.3),
        ],
    }

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize error recovery engine.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.recovery_history: List[Dict[str, Any]] = []

    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """
        Classify an error into a category.

        Args:
            error: The exception that occurred
            context: Additional context about the error

        Returns:
            ErrorCategory classification
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        full_text = f"{error_type} {error_str}"

        # Check against patterns
        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_text):
                    return category

        # Check Python exception types
        if isinstance(error, FileNotFoundError):
            return ErrorCategory.FILE_NOT_FOUND
        if isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION_DENIED
        if isinstance(error, ValueError):
            return ErrorCategory.INVALID_INPUT
        if isinstance(error, TypeError):
            return ErrorCategory.INVALID_INPUT
        if isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        if isinstance(error, MemoryError):
            return ErrorCategory.MEMORY_ERROR

        return ErrorCategory.UNKNOWN

    def analyze_error(
        self,
        error: Exception,
        task: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorAnalysis:
        """
        Analyze an error and generate recovery options.

        Args:
            error: The exception that occurred
            task: The task that was being performed
            context: Additional context

        Returns:
            ErrorAnalysis with recovery options
        """
        category = self.classify_error(error, context)
        context = context or {}
        context["task"] = task

        # Get recovery strategies for this category
        strategies = self.RECOVERY_STRATEGIES.get(category, self.RECOVERY_STRATEGIES[ErrorCategory.UNKNOWN])

        recovery_actions = []
        for strategy, description, confidence in strategies:
            action = RecoveryAction(
                strategy=strategy,
                description=description,
                confidence=confidence,
                parameters=self._build_recovery_params(strategy, error, context),
                requires_user_confirmation=strategy == RecoveryStrategy.REQUEST_USER_INPUT,
            )
            recovery_actions.append(action)

        # Adjust confidence based on context
        recovery_actions = self._adjust_confidence(recovery_actions, error, context)

        # Sort by confidence
        recovery_actions.sort(key=lambda a: a.confidence, reverse=True)

        is_recoverable = any(
            action.strategy != RecoveryStrategy.ABORT for action in recovery_actions
        )

        return ErrorAnalysis(
            category=category,
            message=str(error),
            original_error=error,
            context=context,
            recovery_actions=recovery_actions,
            is_recoverable=is_recoverable,
        )

    def _build_recovery_params(
        self,
        strategy: RecoveryStrategy,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build parameters for a recovery action."""
        params: Dict[str, Any] = {}

        if strategy == RecoveryStrategy.RETRY_SAME:
            params["delay"] = self.base_delay
            params["max_attempts"] = self.max_retries

        elif strategy == RecoveryStrategy.RETRY_MODIFIED:
            # Suggest modifications based on error
            if "timeout" in str(error).lower():
                params["timeout_multiplier"] = 2.0
            if "memory" in str(error).lower():
                params["reduce_batch_size"] = True

        elif strategy == RecoveryStrategy.ALTERNATIVE_CAPABILITY:
            # Suggest alternatives based on task
            task = context.get("task") or ""
            if task and ("file" in task.lower() or "document" in task.lower()):
                params["alternatives"] = ["document_qa", "file_reader", "text_extractor"]
            elif task and "search" in task.lower():
                params["alternatives"] = ["web_search", "deep_research"]

        elif strategy == RecoveryStrategy.SIMPLIFY_TASK:
            params["decompose"] = True
            params["max_subtasks"] = 3

        return params

    def _adjust_confidence(
        self,
        actions: List[RecoveryAction],
        error: Exception,
        context: Dict[str, Any],
    ) -> List[RecoveryAction]:
        """Adjust confidence scores based on context and history."""
        # Check if we've seen this error before
        error_signature = f"{type(error).__name__}:{str(error)[:50]}"

        for action in actions:
            # Check history for similar recoveries
            for hist in self.recovery_history:
                if hist.get("error_signature") == error_signature:
                    if hist.get("strategy") == action.strategy.value:
                        if hist.get("success"):
                            action.confidence = min(1.0, action.confidence + 0.2)
                        else:
                            action.confidence = max(0.1, action.confidence - 0.2)

        return actions

    def record_recovery_attempt(
        self,
        error: Exception,
        strategy: RecoveryStrategy,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a recovery attempt for learning.

        Args:
            error: The error that was recovered from
            strategy: The strategy that was used
            success: Whether the recovery was successful
            context: Additional context
        """
        record = {
            "error_signature": f"{type(error).__name__}:{str(error)[:50]}",
            "strategy": strategy.value,
            "success": success,
            "context": context or {},
        }
        self.recovery_history.append(record)

        # Keep history bounded
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]

        logger.info(f"Recorded recovery attempt: {strategy.value} -> {'success' if success else 'failed'}")

    def format_error_for_user(self, analysis: ErrorAnalysis) -> str:
        """
        Format an error analysis into a user-friendly message.

        Args:
            analysis: The error analysis

        Returns:
            Formatted error message with suggestions
        """
        lines = []
        lines.append(f"Error: {analysis.message}")
        lines.append(f"Category: {analysis.category.value}")
        lines.append("")

        if analysis.is_recoverable:
            lines.append("Suggested recovery options:")
            for i, action in enumerate(analysis.recovery_actions[:3], 1):
                confidence_bar = "=" * int(action.confidence * 10)
                lines.append(f"  {i}. {action.description} [{confidence_bar}] ({action.confidence:.0%})")
        else:
            lines.append("This error cannot be automatically recovered.")
            lines.append("Please check the error details and try again.")

        return "\n".join(lines)

    async def attempt_recovery(
        self,
        analysis: ErrorAnalysis,
        retry_func: Callable,
        **kwargs: Any,
    ) -> Tuple[bool, Any]:
        """
        Attempt to recover from an error.

        Args:
            analysis: Error analysis with recovery options
            retry_func: Function to retry
            **kwargs: Arguments to pass to retry function

        Returns:
            Tuple of (success, result_or_error)
        """
        import asyncio

        best_action = analysis.get_best_action()
        if not best_action:
            return False, analysis.original_error

        logger.info(f"Attempting recovery: {best_action}")

        if best_action.strategy == RecoveryStrategy.RETRY_SAME:
            delay = best_action.parameters.get("delay", self.base_delay)
            max_attempts = best_action.parameters.get("max_attempts", self.max_retries)

            for attempt in range(max_attempts):
                try:
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    result = await retry_func(**kwargs)
                    self.record_recovery_attempt(
                        analysis.original_error, best_action.strategy, True
                    )
                    return True, result
                except Exception as e:
                    if attempt == max_attempts - 1:
                        self.record_recovery_attempt(
                            analysis.original_error, best_action.strategy, False
                        )
                        return False, e

        elif best_action.strategy == RecoveryStrategy.RETRY_MODIFIED:
            # Apply modifications from parameters
            modified_kwargs = kwargs.copy()
            if best_action.parameters.get("timeout_multiplier"):
                if "timeout" in modified_kwargs:
                    modified_kwargs["timeout"] *= best_action.parameters["timeout_multiplier"]

            try:
                result = await retry_func(**modified_kwargs)
                self.record_recovery_attempt(
                    analysis.original_error, best_action.strategy, True
                )
                return True, result
            except Exception as e:
                self.record_recovery_attempt(
                    analysis.original_error, best_action.strategy, False
                )
                return False, e

        elif best_action.strategy == RecoveryStrategy.FALLBACK_RESPONSE:
            # Return a fallback response
            fallback = {
                "status": "partial",
                "message": "Using fallback response due to error",
                "original_error": str(analysis.original_error),
            }
            return True, fallback

        # For other strategies that need external handling
        return False, analysis.original_error
