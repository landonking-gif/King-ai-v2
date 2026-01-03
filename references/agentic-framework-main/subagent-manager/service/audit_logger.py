"""
Audit Logger for Enterprise Agent Compliance Logging.

Module: subagent-manager/service/audit_logger.py

Provides comprehensive audit logging for enterprise agent operations,
supporting compliance requirements and forensic analysis.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AuditPhase(str, Enum):
    """Phases of enterprise agent execution for audit logging."""

    START = "start"
    THINK = "think"
    PLAN = "plan"
    APPROVE = "approve"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REFLECT = "reflect"
    COMPLETE = "complete"
    ERROR = "error"

    # Additional phases for detailed tracking
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    HUMAN_INTERACTION = "human_interaction"
    POLICY_CHECK = "policy_check"
    RETRY = "retry"
    ABORT = "abort"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """
    A single audit log event.

    Captures all relevant information for compliance and forensic analysis.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Event classification
    phase: AuditPhase = Field(...)
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)

    # Execution context
    execution_id: Optional[str] = Field(None, description="Execution context ID")
    task: Optional[str] = Field(None, description="Task being executed")
    user_id: Optional[str] = Field(None, description="User who initiated")

    # Phase-specific fields (all optional, used based on phase)
    # THINK phase
    reasoning_trace: Optional[str] = Field(None)

    # PLAN phase
    plan_id: Optional[str] = Field(None)
    confidence: Optional[float] = Field(None)
    risk_level: Optional[str] = Field(None)
    steps_count: Optional[int] = Field(None)

    # APPROVE phase
    approved: Optional[bool] = Field(None)
    approver: Optional[str] = Field(None)
    violations: Optional[List[str]] = Field(None)

    # EXECUTE phase
    steps_completed: Optional[int] = Field(None)
    steps_failed: Optional[int] = Field(None)
    success_rate: Optional[float] = Field(None)
    provenance_chain: Optional[List[str]] = Field(None)

    # VALIDATE phase
    is_valid: Optional[bool] = Field(None)
    score: Optional[float] = Field(None)
    criteria_met: Optional[Dict[str, bool]] = Field(None)
    evidence: Optional[List[Dict[str, Any]]] = Field(None)

    # REFLECT phase
    lessons_learned: Optional[List[str]] = Field(None)
    should_retry: Optional[bool] = Field(None)
    refinement_action: Optional[str] = Field(None)

    # COMPLETE phase
    success: Optional[bool] = Field(None)
    blocked: Optional[bool] = Field(None)
    iterations_used: Optional[int] = Field(None)
    total_duration_ms: Optional[float] = Field(None)
    final_phase: Optional[str] = Field(None)

    # ERROR phase
    error: Optional[str] = Field(None)
    error_type: Optional[str] = Field(None)
    stack_trace: Optional[str] = Field(None)

    # TOOL_CALL phase
    tool_name: Optional[str] = Field(None)
    tool_inputs_hash: Optional[str] = Field(None)
    tool_outputs_hash: Optional[str] = Field(None)
    tool_duration_ms: Optional[float] = Field(None)

    # Generic metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = Field(None)


class AuditLogEntry(BaseModel):
    """Formatted audit log entry for storage."""

    event: AuditEvent = Field(...)
    formatted_message: str = Field(...)
    tags: List[str] = Field(default_factory=list)
    searchable_text: str = Field(default="")


class AuditSink:
    """Base class for audit log sinks."""

    async def write(self, entry: AuditLogEntry) -> None:
        """Write an audit entry to the sink."""
        raise NotImplementedError

    async def flush(self) -> None:
        """Flush pending entries."""
        pass

    async def close(self) -> None:
        """Close the sink."""
        pass


class ConsoleAuditSink(AuditSink):
    """Audit sink that writes to console/stdout."""

    def __init__(self, format_json: bool = False):
        self.format_json = format_json

    async def write(self, entry: AuditLogEntry) -> None:
        if self.format_json:
            print(json.dumps(entry.event.model_dump(), default=str))
        else:
            print(f"[AUDIT] {entry.formatted_message}")


class FileAuditSink(AuditSink):
    """Audit sink that writes to a file."""

    def __init__(
        self,
        log_path: Union[str, Path],
        format_json: bool = True,
        rotate_size_mb: int = 100,
    ):
        self.log_path = Path(log_path)
        self.format_json = format_json
        self.rotate_size_mb = rotate_size_mb
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, entry: AuditLogEntry) -> None:
        # Check for rotation
        if self.log_path.exists():
            size_mb = self.log_path.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotate_size_mb:
                self._rotate()

        # Write entry
        with open(self.log_path, "a") as f:
            if self.format_json:
                f.write(json.dumps(entry.event.model_dump(), default=str) + "\n")
            else:
                f.write(entry.formatted_message + "\n")

    def _rotate(self) -> None:
        """Rotate log file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated_path = self.log_path.with_suffix(f".{timestamp}.log")
        self.log_path.rename(rotated_path)
        logger.info(f"Rotated audit log to {rotated_path}")


class MemoryAuditSink(AuditSink):
    """Audit sink that stores in memory (for testing/development)."""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries: List[AuditLogEntry] = []

    async def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_entries(
        self,
        execution_id: Optional[str] = None,
        phase: Optional[AuditPhase] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> List[AuditLogEntry]:
        """Query stored entries."""
        result = self.entries

        if execution_id:
            result = [e for e in result if e.event.execution_id == execution_id]
        if phase:
            result = [e for e in result if e.event.phase == phase]
        if severity:
            result = [e for e in result if e.event.severity == severity]

        return result

    def clear(self) -> None:
        """Clear stored entries."""
        self.entries.clear()


class AuditLogger:
    """
    Comprehensive audit logger for enterprise agent operations.

    Supports multiple sinks (console, file, external systems) and
    provides structured logging with full context capture.
    """

    def __init__(
        self,
        sinks: Optional[List[AuditSink]] = None,
        default_severity: AuditSeverity = AuditSeverity.INFO,
        include_metadata: bool = True,
        redact_sensitive: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            sinks: List of audit sinks (default: memory sink)
            default_severity: Default severity for events
            include_metadata: Whether to include metadata in logs
            redact_sensitive: Whether to redact sensitive data
        """
        self.sinks = sinks or [MemoryAuditSink()]
        self.default_severity = default_severity
        self.include_metadata = include_metadata
        self.redact_sensitive = redact_sensitive

        # Event hooks for external integrations
        self._hooks: List[Callable[[AuditEvent], None]] = []

        # Statistics
        self._event_counts: Dict[str, int] = {}

    def log_event(self, event: AuditEvent) -> str:
        """
        Log an audit event.

        Args:
            event: Event to log

        Returns:
            Event ID
        """
        # Apply redaction if enabled
        if self.redact_sensitive:
            event = self._redact_event(event)

        # Format message
        formatted = self._format_event(event)

        # Create entry
        entry = AuditLogEntry(
            event=event,
            formatted_message=formatted,
            tags=self._generate_tags(event),
            searchable_text=self._generate_searchable(event),
        )

        # Write to all sinks
        import asyncio
        for sink in self.sinks:
            try:
                asyncio.create_task(sink.write(entry))
            except RuntimeError:
                # Not in async context, use sync approach
                self._sync_write(sink, entry)

        # Update statistics
        phase_key = event.phase.value
        self._event_counts[phase_key] = self._event_counts.get(phase_key, 0) + 1

        # Call hooks
        for hook in self._hooks:
            try:
                hook(event)
            except Exception as e:
                logger.warning(f"Audit hook failed: {e}")

        logger.debug(f"Audit logged: {event.event_id} ({event.phase.value})")

        return event.event_id

    def _sync_write(self, sink: AuditSink, entry: AuditLogEntry) -> None:
        """Synchronous write fallback."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(sink.write(entry))
            else:
                loop.run_until_complete(sink.write(entry))
        except Exception:
            # Last resort - direct call if coroutine can be awaited
            pass

    def _format_event(self, event: AuditEvent) -> str:
        """Format event as human-readable message."""
        ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        phase = event.phase.value.upper()
        severity = event.severity.value.upper()

        # Build message based on phase
        msg_parts = [f"{ts} [{severity}] [{phase}]"]

        if event.execution_id:
            msg_parts.append(f"exec={event.execution_id[:8]}")

        # Phase-specific content
        if event.phase == AuditPhase.START:
            msg_parts.append(f"task=\"{event.task[:50]}...\"" if event.task else "")
            if event.user_id:
                msg_parts.append(f"user={event.user_id}")

        elif event.phase == AuditPhase.THINK:
            if event.duration_ms:
                msg_parts.append(f"duration={event.duration_ms:.0f}ms")
            if event.reasoning_trace:
                msg_parts.append(f"reasoning={len(event.reasoning_trace)}chars")

        elif event.phase == AuditPhase.PLAN:
            if event.plan_id:
                msg_parts.append(f"plan={event.plan_id[:8]}")
            if event.confidence is not None:
                msg_parts.append(f"confidence={event.confidence:.2f}")
            if event.risk_level:
                msg_parts.append(f"risk={event.risk_level}")
            if event.steps_count:
                msg_parts.append(f"steps={event.steps_count}")

        elif event.phase == AuditPhase.APPROVE:
            msg_parts.append(f"approved={event.approved}")
            if event.approver:
                msg_parts.append(f"by={event.approver}")
            if event.violations:
                msg_parts.append(f"violations={len(event.violations)}")

        elif event.phase == AuditPhase.EXECUTE:
            if event.steps_completed is not None:
                msg_parts.append(f"completed={event.steps_completed}")
            if event.steps_failed is not None:
                msg_parts.append(f"failed={event.steps_failed}")
            if event.success_rate is not None:
                msg_parts.append(f"success_rate={event.success_rate:.2%}")

        elif event.phase == AuditPhase.VALIDATE:
            msg_parts.append(f"valid={event.is_valid}")
            if event.score is not None:
                msg_parts.append(f"score={event.score:.2f}")

        elif event.phase == AuditPhase.REFLECT:
            msg_parts.append(f"should_retry={event.should_retry}")
            if event.refinement_action:
                msg_parts.append(f"action={event.refinement_action}")

        elif event.phase == AuditPhase.COMPLETE:
            msg_parts.append(f"success={event.success}")
            if event.blocked:
                msg_parts.append("BLOCKED")
            if event.iterations_used:
                msg_parts.append(f"iterations={event.iterations_used}")
            if event.total_duration_ms:
                msg_parts.append(f"total_duration={event.total_duration_ms:.0f}ms")

        elif event.phase == AuditPhase.ERROR:
            msg_parts.append(f"error=\"{event.error}\"")
            if event.error_type:
                msg_parts.append(f"type={event.error_type}")

        elif event.phase == AuditPhase.TOOL_CALL:
            if event.tool_name:
                msg_parts.append(f"tool={event.tool_name}")
            if event.tool_duration_ms:
                msg_parts.append(f"duration={event.tool_duration_ms:.0f}ms")

        return " ".join(filter(None, msg_parts))

    def _generate_tags(self, event: AuditEvent) -> List[str]:
        """Generate searchable tags for event."""
        tags = [event.phase.value, event.severity.value]

        if event.execution_id:
            tags.append(f"exec:{event.execution_id}")
        if event.user_id:
            tags.append(f"user:{event.user_id}")
        if event.risk_level:
            tags.append(f"risk:{event.risk_level}")
        if event.approved is not None:
            tags.append("approved" if event.approved else "denied")
        if event.success is not None:
            tags.append("success" if event.success else "failure")
        if event.blocked:
            tags.append("blocked")
        if event.tool_name:
            tags.append(f"tool:{event.tool_name}")

        return tags

    def _generate_searchable(self, event: AuditEvent) -> str:
        """Generate searchable text from event."""
        parts = []

        if event.task:
            parts.append(event.task)
        if event.reasoning_trace:
            parts.append(event.reasoning_trace[:500])
        if event.error:
            parts.append(event.error)
        if event.violations:
            parts.extend(event.violations)
        if event.lessons_learned:
            parts.extend(event.lessons_learned)

        return " ".join(parts)

    def _redact_event(self, event: AuditEvent) -> AuditEvent:
        """Redact sensitive information from event."""
        import re

        sensitive_patterns = [
            (r"api[_-]?key[\"']?\s*[:=]\s*[\"']?[\w-]+", "api_key=***REDACTED***"),
            (r"password[\"']?\s*[:=]\s*[\"']?\S+", "password=***REDACTED***"),
            (r"secret[\"']?\s*[:=]\s*[\"']?\S+", "secret=***REDACTED***"),
            (r"token[\"']?\s*[:=]\s*[\"']?[\w-]+", "token=***REDACTED***"),
        ]

        # Clone event
        event_data = event.model_dump()

        # Redact string fields
        for key, value in event_data.items():
            if isinstance(value, str):
                for pattern, replacement in sensitive_patterns:
                    value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
                event_data[key] = value

        return AuditEvent(**event_data)

    def add_hook(self, hook: Callable[[AuditEvent], None]) -> None:
        """Add a hook to be called for each event."""
        self._hooks.append(hook)

    def remove_hook(self, hook: Callable[[AuditEvent], None]) -> bool:
        """Remove a hook."""
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def add_sink(self, sink: AuditSink) -> None:
        """Add an audit sink."""
        self.sinks.append(sink)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        return {
            "event_counts": self._event_counts.copy(),
            "total_events": sum(self._event_counts.values()),
            "sinks": len(self.sinks),
            "hooks": len(self._hooks),
        }

    def get_events_by_execution(
        self, execution_id: str
    ) -> List[AuditEvent]:
        """Get all events for an execution (from memory sink)."""
        for sink in self.sinks:
            if isinstance(sink, MemoryAuditSink):
                entries = sink.get_entries(execution_id=execution_id)
                return [e.event for e in entries]
        return []

    def get_events_by_phase(
        self, phase: AuditPhase
    ) -> List[AuditEvent]:
        """Get all events for a phase (from memory sink)."""
        for sink in self.sinks:
            if isinstance(sink, MemoryAuditSink):
                entries = sink.get_entries(phase=phase)
                return [e.event for e in entries]
        return []

    async def flush(self) -> None:
        """Flush all sinks."""
        for sink in self.sinks:
            await sink.flush()

    async def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            await sink.close()


# Convenience function for quick logging
def create_audit_event(
    phase: AuditPhase,
    execution_id: Optional[str] = None,
    **kwargs: Any,
) -> AuditEvent:
    """
    Create an audit event with convenience defaults.

    Args:
        phase: Event phase
        execution_id: Optional execution ID
        **kwargs: Additional event fields

    Returns:
        AuditEvent
    """
    return AuditEvent(
        phase=phase,
        execution_id=execution_id,
        **kwargs,
    )
