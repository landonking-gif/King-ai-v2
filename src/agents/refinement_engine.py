"""
Refinement Engine.

Analyzes failures to determine optimal recovery strategy.
Based on agentic-framework refinement patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from src.utils.structured_logging import get_logger

logger = get_logger("refinement_engine")


class FailureType(str, Enum):
    """Types of failures."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    LOGIC_ERROR = "logic_error"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_NOT_FOUND = "resource_not_found"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_RESPONSE = "invalid_response"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    MODIFY_INPUT = "modify_input"
    SIMPLIFY_REQUEST = "simplify_request"
    DECOMPOSE = "decompose"
    USE_FALLBACK = "use_fallback"
    SWITCH_MODEL = "switch_model"
    ESCALATE = "escalate"
    SKIP = "skip"
    ABORT = "abort"


class FailureSeverity(str, Enum):
    """Severity of a failure."""
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    SERIOUS = "serious"
    FATAL = "fatal"


@dataclass
class FailureAnalysis:
    """Analysis of a failure."""
    failure_type: FailureType
    severity: FailureSeverity
    is_transient: bool
    error_message: str
    error_code: Optional[str] = None
    root_cause: Optional[str] = None
    suggested_strategies: List[RecoveryStrategy] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "is_transient": self.is_transient,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "root_cause": self.root_cause,
            "suggested_strategies": [s.value for s in self.suggested_strategies],
            "context": self.context,
        }


@dataclass
class RecoveryAction:
    """A recovery action to take."""
    strategy: RecoveryStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "parameters": self.parameters,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class RefinementResult:
    """Result of a refinement attempt."""
    id: str = field(default_factory=lambda: f"ref_{uuid4().hex[:8]}")
    original_error: str = ""
    analysis: Optional[FailureAnalysis] = None
    action_taken: Optional[RecoveryAction] = None
    success: bool = False
    new_error: Optional[str] = None
    attempt_number: int = 1
    duration_ms: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_error": self.original_error,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "action_taken": self.action_taken.to_dict() if self.action_taken else None,
            "success": self.success,
            "new_error": self.new_error,
            "attempt_number": self.attempt_number,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
        }


class RefinementEngine:
    """
    Analyzes failures and determines recovery strategies.
    
    Features:
    - Failure classification
    - Root cause analysis
    - Strategy selection
    - Automatic recovery
    - Learning from past failures
    """
    
    # Error patterns and their classifications
    ERROR_PATTERNS = {
        # Transient errors - typically recoverable with retry
        FailureType.TIMEOUT: {
            "patterns": ["timeout", "timed out", "deadline exceeded", "context deadline"],
            "severity": FailureSeverity.TRANSIENT,
            "strategies": [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.SWITCH_MODEL],
        },
        FailureType.RATE_LIMIT: {
            "patterns": ["rate limit", "too many requests", "429", "quota exceeded"],
            "severity": FailureSeverity.TRANSIENT,
            "strategies": [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.SWITCH_MODEL],
        },
        FailureType.NETWORK_ERROR: {
            "patterns": ["connection refused", "network unreachable", "dns", "socket"],
            "severity": FailureSeverity.TRANSIENT,
            "strategies": [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.USE_FALLBACK],
        },
        
        # Recoverable errors - may need input modification
        FailureType.VALIDATION_ERROR: {
            "patterns": ["validation", "invalid", "malformed", "schema"],
            "severity": FailureSeverity.RECOVERABLE,
            "strategies": [RecoveryStrategy.MODIFY_INPUT, RecoveryStrategy.SIMPLIFY_REQUEST],
        },
        FailureType.INVALID_RESPONSE: {
            "patterns": ["json", "parse", "unexpected format", "cannot decode"],
            "severity": FailureSeverity.RECOVERABLE,
            "strategies": [RecoveryStrategy.RETRY, RecoveryStrategy.SWITCH_MODEL],
        },
        FailureType.RESOURCE_NOT_FOUND: {
            "patterns": ["not found", "404", "does not exist", "missing"],
            "severity": FailureSeverity.RECOVERABLE,
            "strategies": [RecoveryStrategy.MODIFY_INPUT, RecoveryStrategy.SKIP],
        },
        
        # Serious errors - need escalation or major changes
        FailureType.API_ERROR: {
            "patterns": ["api error", "500", "internal server error", "service unavailable"],
            "severity": FailureSeverity.SERIOUS,
            "strategies": [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.USE_FALLBACK, RecoveryStrategy.ESCALATE],
        },
        FailureType.QUOTA_EXCEEDED: {
            "patterns": ["quota", "billing", "exceeded limit", "payment"],
            "severity": FailureSeverity.SERIOUS,
            "strategies": [RecoveryStrategy.SWITCH_MODEL, RecoveryStrategy.ESCALATE],
        },
        FailureType.LOGIC_ERROR: {
            "patterns": ["assertion", "invariant", "unexpected state", "logic error"],
            "severity": FailureSeverity.SERIOUS,
            "strategies": [RecoveryStrategy.DECOMPOSE, RecoveryStrategy.ESCALATE],
        },
        
        # Fatal errors - cannot recover automatically
        FailureType.PERMISSION_DENIED: {
            "patterns": ["permission denied", "403", "unauthorized", "forbidden", "access denied"],
            "severity": FailureSeverity.FATAL,
            "strategies": [RecoveryStrategy.ESCALATE, RecoveryStrategy.ABORT],
        },
        FailureType.DEPENDENCY_FAILED: {
            "patterns": ["dependency", "prerequisite", "required step failed"],
            "severity": FailureSeverity.FATAL,
            "strategies": [RecoveryStrategy.ABORT],
        },
    }
    
    # Backoff parameters by failure type
    BACKOFF_PARAMS = {
        FailureType.TIMEOUT: {"base_delay": 1.0, "max_delay": 30.0, "multiplier": 2.0},
        FailureType.RATE_LIMIT: {"base_delay": 5.0, "max_delay": 60.0, "multiplier": 2.0},
        FailureType.NETWORK_ERROR: {"base_delay": 2.0, "max_delay": 30.0, "multiplier": 1.5},
        FailureType.API_ERROR: {"base_delay": 3.0, "max_delay": 60.0, "multiplier": 2.0},
    }
    
    def __init__(
        self,
        max_retries: int = 3,
        fallback_handler: Callable[[str, Dict[str, Any]], Any] = None,
        escalation_handler: Callable[[FailureAnalysis], None] = None,
    ):
        """
        Initialize the engine.
        
        Args:
            max_retries: Maximum retry attempts
            fallback_handler: Handler for fallback strategy
            escalation_handler: Handler for escalation
        """
        self.max_retries = max_retries
        self.fallback_handler = fallback_handler
        self.escalation_handler = escalation_handler
        
        # Track failure history for learning
        self._failure_history: List[RefinementResult] = []
        self._strategy_success: Dict[RecoveryStrategy, Dict[str, int]] = {
            strategy: {"attempts": 0, "successes": 0}
            for strategy in RecoveryStrategy
        }
    
    def analyze_failure(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
    ) -> FailureAnalysis:
        """
        Analyze a failure to determine type and recovery options.
        
        Args:
            error: The error/exception
            context: Additional context about the failure
            
        Returns:
            Failure analysis
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        context = context or {}
        
        # Try to match error patterns
        failure_type = FailureType.UNKNOWN
        severity = FailureSeverity.RECOVERABLE
        strategies = []
        
        for ftype, info in self.ERROR_PATTERNS.items():
            for pattern in info["patterns"]:
                if pattern in error_str or pattern in error_type.lower():
                    failure_type = ftype
                    severity = info["severity"]
                    strategies = info["strategies"]
                    break
            if failure_type != FailureType.UNKNOWN:
                break
        
        # Default strategies if unknown
        if not strategies:
            strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE]
        
        # Determine if transient
        is_transient = severity in [FailureSeverity.TRANSIENT]
        
        # Try to extract root cause
        root_cause = self._extract_root_cause(error, failure_type)
        
        return FailureAnalysis(
            failure_type=failure_type,
            severity=severity,
            is_transient=is_transient,
            error_message=str(error),
            error_code=getattr(error, 'code', None),
            root_cause=root_cause,
            suggested_strategies=strategies,
            context=context,
        )
    
    def select_strategy(
        self,
        analysis: FailureAnalysis,
        attempt_number: int = 1,
        previous_strategies: List[RecoveryStrategy] = None,
    ) -> RecoveryAction:
        """
        Select the best recovery strategy.
        
        Args:
            analysis: Failure analysis
            attempt_number: Current attempt number
            previous_strategies: Strategies already tried
            
        Returns:
            Selected recovery action
        """
        previous_strategies = previous_strategies or []
        
        # Filter out already-tried strategies
        available = [
            s for s in analysis.suggested_strategies
            if s not in previous_strategies
        ]
        
        # If no strategies left, escalate or abort
        if not available:
            if analysis.severity == FailureSeverity.FATAL:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ABORT,
                    reason="No recovery strategies available and failure is fatal",
                    confidence=1.0,
                )
            else:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    reason="All recovery strategies exhausted",
                    confidence=1.0,
                )
        
        # Too many attempts - escalate
        if attempt_number > self.max_retries:
            if RecoveryStrategy.ESCALATE in available:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    reason=f"Maximum retries ({self.max_retries}) exceeded",
                    confidence=1.0,
                )
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                reason=f"Maximum retries ({self.max_retries}) exceeded",
                confidence=1.0,
            )
        
        # Select best strategy based on success history
        best_strategy = available[0]
        best_success_rate = 0.0
        
        for strategy in available:
            stats = self._strategy_success[strategy]
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_strategy = strategy
        
        # Build parameters for the strategy
        params = self._build_strategy_params(
            best_strategy, analysis, attempt_number
        )
        
        return RecoveryAction(
            strategy=best_strategy,
            parameters=params,
            reason=f"Selected based on failure type {analysis.failure_type.value}",
            confidence=max(0.3, best_success_rate) if best_success_rate > 0 else 0.7,
        )
    
    async def execute_recovery(
        self,
        action: RecoveryAction,
        retry_func: Callable,
        original_args: Tuple = None,
        original_kwargs: Dict[str, Any] = None,
    ) -> Tuple[bool, Any]:
        """
        Execute a recovery action.
        
        Args:
            action: Recovery action to execute
            retry_func: Function to retry
            original_args: Original function arguments
            original_kwargs: Original function keyword arguments
            
        Returns:
            Tuple of (success, result)
        """
        import asyncio
        
        original_args = original_args or ()
        original_kwargs = original_kwargs or {}
        
        try:
            if action.strategy == RecoveryStrategy.RETRY:
                result = await retry_func(*original_args, **original_kwargs)
                return True, result
                
            elif action.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                delay = action.parameters.get("delay", 1.0)
                await asyncio.sleep(delay)
                result = await retry_func(*original_args, **original_kwargs)
                return True, result
                
            elif action.strategy == RecoveryStrategy.SIMPLIFY_REQUEST:
                # Simplify by reducing complexity hints
                simplified_kwargs = dict(original_kwargs)
                if "max_tokens" in simplified_kwargs:
                    simplified_kwargs["max_tokens"] = min(500, simplified_kwargs["max_tokens"])
                result = await retry_func(*original_args, **simplified_kwargs)
                return True, result
                
            elif action.strategy == RecoveryStrategy.MODIFY_INPUT:
                # This would need task-specific modification
                modified_kwargs = action.parameters.get("modified_input", original_kwargs)
                result = await retry_func(*original_args, **modified_kwargs)
                return True, result
                
            elif action.strategy == RecoveryStrategy.SWITCH_MODEL:
                # Switch to a different model
                new_model = action.parameters.get("fallback_model", "llama3.2")
                modified_kwargs = dict(original_kwargs)
                modified_kwargs["model"] = new_model
                result = await retry_func(*original_args, **modified_kwargs)
                return True, result
                
            elif action.strategy == RecoveryStrategy.USE_FALLBACK:
                if self.fallback_handler:
                    result = await self.fallback_handler(
                        action.parameters.get("task_type", "unknown"),
                        original_kwargs
                    )
                    return True, result
                return False, None
                
            elif action.strategy == RecoveryStrategy.ESCALATE:
                if self.escalation_handler:
                    await self.escalation_handler(action.parameters.get("analysis"))
                return False, None
                
            elif action.strategy == RecoveryStrategy.SKIP:
                return True, None  # Skip returns success but no result
                
            elif action.strategy == RecoveryStrategy.ABORT:
                return False, None
                
            elif action.strategy == RecoveryStrategy.DECOMPOSE:
                # Would need task-specific decomposition
                return False, None
                
        except Exception as e:
            logger.warning(f"Recovery strategy {action.strategy.value} failed: {e}")
            return False, None
        
        return False, None
    
    def record_result(
        self,
        analysis: FailureAnalysis,
        action: RecoveryAction,
        success: bool,
    ) -> None:
        """Record the result of a recovery attempt for learning."""
        result = RefinementResult(
            original_error=analysis.error_message,
            analysis=analysis,
            action_taken=action,
            success=success,
        )
        
        self._failure_history.append(result)
        
        # Update strategy statistics
        stats = self._strategy_success[action.strategy]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        strategy_stats = {}
        for strategy, stats in self._strategy_success.items():
            rate = stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0
            strategy_stats[strategy.value] = {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": rate,
            }
        
        # Count by failure type
        failure_counts = {}
        for result in self._failure_history:
            if result.analysis:
                ftype = result.analysis.failure_type.value
                failure_counts[ftype] = failure_counts.get(ftype, 0) + 1
        
        return {
            "total_failures": len(self._failure_history),
            "by_failure_type": failure_counts,
            "strategy_stats": strategy_stats,
        }
    
    # Private methods
    
    def _extract_root_cause(
        self,
        error: Exception,
        failure_type: FailureType,
    ) -> Optional[str]:
        """Try to extract root cause from error."""
        error_str = str(error)
        
        if failure_type == FailureType.VALIDATION_ERROR:
            # Try to find the specific field
            if "field" in error_str.lower():
                return f"Input validation failed: {error_str[:100]}"
            return "Input validation failed"
        
        if failure_type == FailureType.TIMEOUT:
            return "Operation timed out - may need longer timeout or simpler request"
        
        if failure_type == FailureType.RATE_LIMIT:
            return "API rate limit reached - need to slow down or use different API"
        
        if failure_type == FailureType.PERMISSION_DENIED:
            return "Access denied - check credentials and permissions"
        
        return None
    
    def _build_strategy_params(
        self,
        strategy: RecoveryStrategy,
        analysis: FailureAnalysis,
        attempt_number: int,
    ) -> Dict[str, Any]:
        """Build parameters for a recovery strategy."""
        params = {}
        
        if strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            backoff = self.BACKOFF_PARAMS.get(
                analysis.failure_type,
                {"base_delay": 1.0, "max_delay": 30.0, "multiplier": 2.0}
            )
            delay = min(
                backoff["max_delay"],
                backoff["base_delay"] * (backoff["multiplier"] ** (attempt_number - 1))
            )
            params["delay"] = delay
        
        if strategy == RecoveryStrategy.SWITCH_MODEL:
            # Default fallback models
            params["fallback_model"] = "llama3.2"  # Local Ollama model
            
        if strategy == RecoveryStrategy.ESCALATE:
            params["analysis"] = analysis
        
        return params


# Global engine instance
_refinement_engine: Optional[RefinementEngine] = None


def get_refinement_engine() -> RefinementEngine:
    """Get or create the global refinement engine."""
    global _refinement_engine
    if _refinement_engine is None:
        _refinement_engine = RefinementEngine()
    return _refinement_engine
