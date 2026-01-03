"""
Session Memory for Kautilya.

Provides in-session learning and context management:
- Stores conversation history and context
- Tracks successful and failed operations
- Learns from recoveries and user corrections
- Maintains working memory for complex tasks
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries."""

    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    RECOVERY = "recovery"
    USER_FEEDBACK = "user_feedback"
    CONTEXT = "context"
    FILE_RESOLUTION = "file_resolution"
    CAPABILITY_USAGE = "capability_usage"


@dataclass
class MemoryEntry:
    """A single entry in session memory."""

    id: str
    type: MemoryType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0  # Decays over time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
        }


@dataclass
class TaskMemory:
    """Memory of a task execution."""

    task_id: str
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    capabilities_used: List[str] = field(default_factory=list)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)


class SessionMemory:
    """
    Session memory manager for learning and context.

    Maintains:
    - Conversation history
    - Task execution history
    - Error and recovery patterns
    - User preferences and corrections
    - File resolution cache
    - Working context for multi-step tasks
    """

    def __init__(
        self,
        max_entries: int = 1000,
        decay_rate: float = 0.95,
        context_window: int = 10,
    ):
        """
        Initialize session memory.

        Args:
            max_entries: Maximum memory entries to keep
            decay_rate: Rate at which relevance decays
            context_window: Number of recent entries to include in context
        """
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.context_window = context_window

        self.entries: List[MemoryEntry] = []
        self.task_memory: Dict[str, TaskMemory] = {}
        self.file_cache: Dict[str, str] = {}  # reference -> resolved path
        self.capability_stats: Dict[str, Dict[str, Any]] = {}
        self.user_corrections: List[Dict[str, Any]] = []
        self.working_context: Dict[str, Any] = {}

    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate a unique memory ID."""
        timestamp = str(time.time())
        hash_input = f"{prefix}_{timestamp}_{len(self.entries)}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def add(
        self,
        type: MemoryType,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an entry to memory.

        Args:
            type: Type of memory entry
            content: Content to store
            metadata: Additional metadata

        Returns:
            Memory entry ID
        """
        entry_id = self._generate_id(type.value)
        entry = MemoryEntry(
            id=entry_id,
            type=type,
            content=content,
            metadata=metadata or {},
        )
        self.entries.append(entry)

        # Apply decay to older entries
        self._apply_decay()

        # Trim if over limit
        if len(self.entries) > self.max_entries:
            self._trim_entries()

        logger.debug(f"Added memory entry: {entry_id} ({type.value})")
        return entry_id

    def _apply_decay(self) -> None:
        """Apply relevance decay to all entries."""
        for entry in self.entries[:-1]:  # Don't decay the newest
            entry.relevance_score *= self.decay_rate

    def _trim_entries(self) -> None:
        """Remove least relevant entries to stay under limit."""
        # Sort by relevance and keep most relevant
        self.entries.sort(key=lambda e: e.relevance_score, reverse=True)
        self.entries = self.entries[: self.max_entries]

    def get_recent(self, count: int = 10, type_filter: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """
        Get recent memory entries.

        Args:
            count: Number of entries to return
            type_filter: Optional filter by type

        Returns:
            List of recent memory entries
        """
        entries = self.entries
        if type_filter:
            entries = [e for e in entries if e.type == type_filter]

        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:count]

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Search memory for relevant entries.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entries
        """
        query_lower = query.lower()
        matches = []

        for entry in self.entries:
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                score = content_str.count(query_lower) * entry.relevance_score
                matches.append((score, entry))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in matches[:limit]]

    # Task Memory Management

    def start_task(self, task_id: str, description: str, inputs: Optional[Dict[str, Any]] = None) -> None:
        """Record the start of a task."""
        self.task_memory[task_id] = TaskMemory(
            task_id=task_id,
            description=description,
            status="in_progress",
            start_time=datetime.now(),
            inputs=inputs or {},
        )

        self.add(
            MemoryType.TASK,
            {"action": "start", "description": description},
            {"task_id": task_id},
        )

    def complete_task(self, task_id: str, outputs: Dict[str, Any]) -> None:
        """Record task completion."""
        if task_id in self.task_memory:
            task = self.task_memory[task_id]
            task.status = "completed"
            task.end_time = datetime.now()
            task.outputs = outputs

            self.add(
                MemoryType.RESULT,
                outputs,
                {"task_id": task_id, "duration": (task.end_time - task.start_time).total_seconds()},
            )

    def fail_task(self, task_id: str, error: str) -> None:
        """Record task failure."""
        if task_id in self.task_memory:
            task = self.task_memory[task_id]
            task.status = "failed"
            task.end_time = datetime.now()
            task.errors.append(error)

            self.add(
                MemoryType.ERROR,
                {"error": error, "task": task.description},
                {"task_id": task_id},
            )

    def record_recovery(self, task_id: str, strategy: str, success: bool) -> None:
        """Record a recovery attempt."""
        if task_id in self.task_memory:
            self.task_memory[task_id].recovery_attempts.append({
                "strategy": strategy,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            })

        self.add(
            MemoryType.RECOVERY,
            {"strategy": strategy, "success": success},
            {"task_id": task_id},
        )

    # File Resolution Cache

    def cache_file_resolution(self, reference: str, resolved_path: str) -> None:
        """Cache a file resolution."""
        self.file_cache[reference] = resolved_path
        self.add(
            MemoryType.FILE_RESOLUTION,
            {"reference": reference, "resolved": resolved_path},
        )

    def get_cached_file(self, reference: str) -> Optional[str]:
        """Get a cached file resolution."""
        return self.file_cache.get(reference)

    # Capability Statistics

    def record_capability_usage(
        self,
        capability_name: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
    ) -> None:
        """Record capability usage for learning."""
        if capability_name not in self.capability_stats:
            self.capability_stats[capability_name] = {
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0,
                "errors": [],
            }

        stats = self.capability_stats[capability_name]
        stats["total_calls"] += 1
        stats["total_duration"] += duration
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            if error:
                stats["errors"].append(error)

        self.add(
            MemoryType.CAPABILITY_USAGE,
            {
                "capability": capability_name,
                "success": success,
                "duration": duration,
            },
        )

    def get_capability_success_rate(self, capability_name: str) -> float:
        """Get success rate for a capability."""
        stats = self.capability_stats.get(capability_name)
        if not stats or stats["total_calls"] == 0:
            return 0.5  # Unknown capability, assume 50%

        return stats["successes"] / stats["total_calls"]

    # User Corrections

    def record_user_correction(
        self,
        original_output: Any,
        correction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record when user corrects the agent."""
        self.user_corrections.append({
            "original": original_output,
            "correction": correction,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        })

        self.add(
            MemoryType.USER_FEEDBACK,
            {"type": "correction", "correction": correction},
        )

    def get_similar_corrections(self, context: str) -> List[Dict[str, Any]]:
        """Find similar past corrections."""
        context_lower = context.lower()
        similar = []

        for correction in self.user_corrections:
            original_str = str(correction.get("original", "")).lower()
            if any(word in original_str for word in context_lower.split()):
                similar.append(correction)

        return similar

    # Working Context

    def set_context(self, key: str, value: Any) -> None:
        """Set a value in working context."""
        self.working_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from working context."""
        return self.working_context.get(key, default)

    def clear_context(self) -> None:
        """Clear working context."""
        self.working_context.clear()

    # Context Building

    def build_context_prompt(self, current_task: str) -> str:
        """
        Build a context prompt from recent memory.

        Args:
            current_task: The current task description

        Returns:
            Context string to include in prompts
        """
        lines = []

        # Recent tasks
        recent_tasks = self.get_recent(count=3, type_filter=MemoryType.TASK)
        if recent_tasks:
            lines.append("Recent tasks:")
            for entry in recent_tasks:
                lines.append(f"  - {entry.content.get('description', 'Unknown')}")

        # Recent errors and recoveries
        recent_errors = self.get_recent(count=2, type_filter=MemoryType.ERROR)
        if recent_errors:
            lines.append("\nRecent issues encountered:")
            for entry in recent_errors:
                lines.append(f"  - {entry.content.get('error', 'Unknown error')[:100]}")

        # Relevant corrections
        similar_corrections = self.get_similar_corrections(current_task)
        if similar_corrections:
            lines.append("\nUser preferences/corrections:")
            for corr in similar_corrections[:2]:
                lines.append(f"  - {corr['correction'][:100]}")

        # Working context
        if self.working_context:
            lines.append("\nCurrent context:")
            for key, value in list(self.working_context.items())[:5]:
                lines.append(f"  - {key}: {str(value)[:50]}")

        return "\n".join(lines) if lines else ""

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of session memory."""
        return {
            "total_entries": len(self.entries),
            "tasks_completed": sum(
                1 for t in self.task_memory.values() if t.status == "completed"
            ),
            "tasks_failed": sum(
                1 for t in self.task_memory.values() if t.status == "failed"
            ),
            "files_cached": len(self.file_cache),
            "capabilities_used": list(self.capability_stats.keys()),
            "user_corrections": len(self.user_corrections),
        }

    def export(self) -> Dict[str, Any]:
        """Export session memory for persistence."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "task_memory": {k: v.__dict__ for k, v in self.task_memory.items()},
            "file_cache": self.file_cache,
            "capability_stats": self.capability_stats,
            "user_corrections": self.user_corrections,
            "working_context": self.working_context,
        }
