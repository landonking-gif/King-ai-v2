"""
Activity Stream.

Structured event logging with taxonomy for dashboard and audit.
Based on agentic-framework activity patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4
from collections import deque
import asyncio
import json

from src.utils.structured_logging import get_logger

logger = get_logger("activity_stream")


class ActivityCategory(str, Enum):
    """High-level activity categories."""
    AGENT = "agent"
    WORKFLOW = "workflow"
    BUSINESS = "business"
    USER = "user"
    SYSTEM = "system"
    APPROVAL = "approval"
    INTEGRATION = "integration"
    ERROR = "error"


class ActivityType(str, Enum):
    """Specific activity types."""
    # Agent activities
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_DELEGATED = "agent.delegated"
    
    # Workflow activities
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    
    # Business activities
    BUSINESS_CREATED = "business.created"
    BUSINESS_UPDATED = "business.updated"
    BUSINESS_PHASE_CHANGED = "business.phase.changed"
    BUSINESS_MILESTONE = "business.milestone"
    
    # User activities
    USER_COMMAND = "user.command"
    USER_QUERY = "user.query"
    USER_FEEDBACK = "user.feedback"
    
    # System activities
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_HEALTH_CHECK = "system.health.check"
    
    # Approval activities
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_ESCALATED = "approval.escalated"
    
    # Integration activities
    INTEGRATION_CALL = "integration.call"
    INTEGRATION_WEBHOOK = "integration.webhook"
    INTEGRATION_ERROR = "integration.error"
    
    # Error activities
    ERROR_VALIDATION = "error.validation"
    ERROR_EXECUTION = "error.execution"
    ERROR_TIMEOUT = "error.timeout"
    ERROR_RATE_LIMIT = "error.rate_limit"


class ActivityLevel(str, Enum):
    """Activity importance level."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Activity:
    """A single activity event."""
    id: str = field(default_factory=lambda: f"act_{uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Classification
    category: ActivityCategory = ActivityCategory.SYSTEM
    activity_type: ActivityType = ActivityType.SYSTEM_HEALTH_CHECK
    level: ActivityLevel = ActivityLevel.INFO
    
    # Context
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    business_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Content
    title: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    duration_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "activity_type": self.activity_type.value,
            "level": self.level.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "business_id": self.business_id,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "title": self.title,
            "description": self.description,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "tags": self.tags,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass 
class ActivityFilter:
    """Filter for querying activities."""
    categories: Optional[Set[ActivityCategory]] = None
    activity_types: Optional[Set[ActivityType]] = None
    levels: Optional[Set[ActivityLevel]] = None
    business_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    actor_id: Optional[str] = None
    tags: Optional[Set[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def matches(self, activity: Activity) -> bool:
        """Check if an activity matches this filter."""
        if self.categories and activity.category not in self.categories:
            return False
        if self.activity_types and activity.activity_type not in self.activity_types:
            return False
        if self.levels and activity.level not in self.levels:
            return False
        if self.business_id and activity.business_id != self.business_id:
            return False
        if self.session_id and activity.session_id != self.session_id:
            return False
        if self.workflow_id and activity.workflow_id != self.workflow_id:
            return False
        if self.actor_id and activity.actor_id != self.actor_id:
            return False
        if self.tags and not self.tags.intersection(set(activity.tags)):
            return False
        if self.start_time and activity.timestamp < self.start_time:
            return False
        if self.end_time and activity.timestamp > self.end_time:
            return False
        return True


class ActivityStream:
    """
    Central activity stream for all system events.
    
    Features:
    - Real-time event streaming
    - Subscriber notifications
    - Filtering and querying
    - Statistics and aggregations
    - Dashboard integration
    """
    
    def __init__(
        self,
        max_buffer_size: int = 10000,
        persist_callback: Optional[Callable[[Activity], None]] = None,
    ):
        """
        Initialize activity stream.
        
        Args:
            max_buffer_size: Maximum activities to keep in memory
            persist_callback: Optional callback to persist activities
        """
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._persist_callback = persist_callback
        self._subscribers: Dict[str, Callable[[Activity], None]] = {}
        self._async_subscribers: Dict[str, Callable[[Activity], Any]] = {}
        
        # Statistics
        self._stats = {
            "total_activities": 0,
            "by_category": {cat.value: 0 for cat in ActivityCategory},
            "by_level": {lvl.value: 0 for lvl in ActivityLevel},
            "last_activity_time": None,
        }
    
    def emit(
        self,
        activity_type: ActivityType,
        title: str,
        description: str = "",
        category: ActivityCategory = None,
        level: ActivityLevel = ActivityLevel.INFO,
        **kwargs,
    ) -> Activity:
        """
        Emit a new activity.
        
        Args:
            activity_type: Type of activity
            title: Short title
            description: Detailed description
            category: Category (auto-detected if not provided)
            level: Importance level
            **kwargs: Additional activity fields
            
        Returns:
            Created activity
        """
        # Auto-detect category from type
        if category is None:
            type_prefix = activity_type.value.split(".")[0]
            category_map = {
                "agent": ActivityCategory.AGENT,
                "workflow": ActivityCategory.WORKFLOW,
                "business": ActivityCategory.BUSINESS,
                "user": ActivityCategory.USER,
                "system": ActivityCategory.SYSTEM,
                "approval": ActivityCategory.APPROVAL,
                "integration": ActivityCategory.INTEGRATION,
                "error": ActivityCategory.ERROR,
            }
            category = category_map.get(type_prefix, ActivityCategory.SYSTEM)
        
        activity = Activity(
            activity_type=activity_type,
            title=title,
            description=description,
            category=category,
            level=level,
            **kwargs,
        )
        
        # Store in buffer
        self._buffer.append(activity)
        
        # Update stats
        self._stats["total_activities"] += 1
        self._stats["by_category"][category.value] += 1
        self._stats["by_level"][level.value] += 1
        self._stats["last_activity_time"] = activity.timestamp
        
        # Persist if callback provided
        if self._persist_callback:
            try:
                self._persist_callback(activity)
            except Exception as e:
                logger.warning(f"Failed to persist activity: {e}")
        
        # Notify subscribers
        self._notify_subscribers(activity)
        
        return activity
    
    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[Activity], None],
        filter: ActivityFilter = None,
    ) -> str:
        """
        Subscribe to activity notifications.
        
        Args:
            subscriber_id: Unique subscriber ID
            callback: Function to call with activities
            filter: Optional filter for activities
            
        Returns:
            Subscriber ID
        """
        # Wrap with filter if provided
        if filter:
            original_callback = callback
            def filtered_callback(activity: Activity):
                if filter.matches(activity):
                    original_callback(activity)
            self._subscribers[subscriber_id] = filtered_callback
        else:
            self._subscribers[subscriber_id] = callback
        
        logger.debug(f"Subscriber registered: {subscriber_id}")
        return subscriber_id
    
    def subscribe_async(
        self,
        subscriber_id: str,
        callback: Callable[[Activity], Any],
        filter: ActivityFilter = None,
    ) -> str:
        """Subscribe with async callback."""
        if filter:
            original_callback = callback
            async def filtered_callback(activity: Activity):
                if filter.matches(activity):
                    await original_callback(activity)
            self._async_subscribers[subscriber_id] = filtered_callback
        else:
            self._async_subscribers[subscriber_id] = callback
        
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from notifications."""
        removed = False
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]
            removed = True
        if subscriber_id in self._async_subscribers:
            del self._async_subscribers[subscriber_id]
            removed = True
        return removed
    
    def query(
        self,
        filter: ActivityFilter = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Activity]:
        """
        Query activities from buffer.
        
        Args:
            filter: Optional filter
            limit: Maximum results
            offset: Skip first N results
            
        Returns:
            Matching activities (newest first)
        """
        # Get from buffer in reverse order (newest first)
        activities = list(reversed(self._buffer))
        
        # Apply filter
        if filter:
            activities = [a for a in activities if filter.matches(a)]
        
        # Apply pagination
        return activities[offset:offset + limit]
    
    def query_recent(
        self,
        minutes: int = 60,
        filter: ActivityFilter = None,
    ) -> List[Activity]:
        """Query activities from the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        activities = [
            a for a in self._buffer
            if a.timestamp >= cutoff
        ]
        
        if filter:
            activities = [a for a in activities if filter.matches(a)]
        
        return list(reversed(activities))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get activity statistics."""
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
            "subscriber_count": len(self._subscribers) + len(self._async_subscribers),
        }
    
    def get_summary(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get activity summary for dashboard.
        
        Args:
            hours: Hours to summarize
            
        Returns:
            Summary statistics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [a for a in self._buffer if a.timestamp >= cutoff]
        
        # Count by category
        by_category = {}
        for cat in ActivityCategory:
            by_category[cat.value] = sum(1 for a in recent if a.category == cat)
        
        # Count by level
        by_level = {}
        for lvl in ActivityLevel:
            by_level[lvl.value] = sum(1 for a in recent if a.level == lvl)
        
        # Recent errors
        errors = [
            a.to_dict() for a in recent
            if a.level in [ActivityLevel.ERROR, ActivityLevel.CRITICAL]
        ][-10:]  # Last 10 errors
        
        # Unique businesses
        businesses = set(a.business_id for a in recent if a.business_id)
        
        # Agent activity
        agent_activities = [a for a in recent if a.category == ActivityCategory.AGENT]
        
        return {
            "period_hours": hours,
            "total_activities": len(recent),
            "by_category": by_category,
            "by_level": by_level,
            "recent_errors": errors,
            "unique_businesses": len(businesses),
            "agent_executions": len(agent_activities),
            "total_cost": sum(a.cost or 0 for a in recent),
            "total_tokens": sum(a.tokens_used or 0 for a in recent),
        }
    
    def clear(self) -> int:
        """Clear all buffered activities. Returns count cleared."""
        count = len(self._buffer)
        self._buffer.clear()
        return count
    
    # Private methods
    
    def _notify_subscribers(self, activity: Activity) -> None:
        """Notify all subscribers of new activity."""
        # Sync subscribers
        for subscriber_id, callback in self._subscribers.items():
            try:
                callback(activity)
            except Exception as e:
                logger.warning(f"Subscriber {subscriber_id} failed: {e}")
        
        # Async subscribers (schedule)
        for subscriber_id, callback in self._async_subscribers.items():
            try:
                # Schedule async callback
                asyncio.create_task(self._notify_async(subscriber_id, callback, activity))
            except RuntimeError:
                # No event loop, skip
                pass
    
    async def _notify_async(
        self,
        subscriber_id: str,
        callback: Callable[[Activity], Any],
        activity: Activity,
    ) -> None:
        """Notify async subscriber."""
        try:
            await callback(activity)
        except Exception as e:
            logger.warning(f"Async subscriber {subscriber_id} failed: {e}")


# Convenience functions for common activity emissions
class ActivityEmitter:
    """Convenience class for emitting typed activities."""
    
    def __init__(self, stream: ActivityStream):
        """Initialize with stream."""
        self.stream = stream
    
    def agent_started(
        self,
        agent_id: str,
        agent_name: str,
        task: str,
        **kwargs,
    ) -> Activity:
        """Emit agent started activity."""
        return self.stream.emit(
            ActivityType.AGENT_STARTED,
            f"Agent started: {agent_name}",
            f"Starting task: {task}",
            actor_id=agent_id,
            actor_type="agent",
            data={"task": task},
            **kwargs,
        )
    
    def agent_completed(
        self,
        agent_id: str,
        agent_name: str,
        duration_ms: int,
        **kwargs,
    ) -> Activity:
        """Emit agent completed activity."""
        return self.stream.emit(
            ActivityType.AGENT_COMPLETED,
            f"Agent completed: {agent_name}",
            f"Completed in {duration_ms}ms",
            actor_id=agent_id,
            actor_type="agent",
            duration_ms=duration_ms,
            **kwargs,
        )
    
    def agent_failed(
        self,
        agent_id: str,
        agent_name: str,
        error: str,
        **kwargs,
    ) -> Activity:
        """Emit agent failed activity."""
        return self.stream.emit(
            ActivityType.AGENT_FAILED,
            f"Agent failed: {agent_name}",
            error,
            level=ActivityLevel.ERROR,
            actor_id=agent_id,
            actor_type="agent",
            data={"error": error},
            **kwargs,
        )
    
    def workflow_started(
        self,
        workflow_id: str,
        workflow_name: str,
        **kwargs,
    ) -> Activity:
        """Emit workflow started activity."""
        return self.stream.emit(
            ActivityType.WORKFLOW_STARTED,
            f"Workflow started: {workflow_name}",
            workflow_id=workflow_id,
            **kwargs,
        )
    
    def workflow_completed(
        self,
        workflow_id: str,
        workflow_name: str,
        duration_ms: int,
        **kwargs,
    ) -> Activity:
        """Emit workflow completed activity."""
        return self.stream.emit(
            ActivityType.WORKFLOW_COMPLETED,
            f"Workflow completed: {workflow_name}",
            f"Completed in {duration_ms}ms",
            workflow_id=workflow_id,
            duration_ms=duration_ms,
            **kwargs,
        )
    
    def business_created(
        self,
        business_id: str,
        business_name: str,
        business_type: str,
        **kwargs,
    ) -> Activity:
        """Emit business created activity."""
        return self.stream.emit(
            ActivityType.BUSINESS_CREATED,
            f"Business created: {business_name}",
            f"Type: {business_type}",
            business_id=business_id,
            data={"business_type": business_type},
            **kwargs,
        )
    
    def approval_requested(
        self,
        operation: str,
        requested_by: str,
        **kwargs,
    ) -> Activity:
        """Emit approval requested activity."""
        return self.stream.emit(
            ActivityType.APPROVAL_REQUESTED,
            f"Approval requested: {operation}",
            f"Requested by: {requested_by}",
            level=ActivityLevel.WARNING,
            actor_id=requested_by,
            **kwargs,
        )
    
    def error(
        self,
        error_type: str,
        message: str,
        **kwargs,
    ) -> Activity:
        """Emit error activity."""
        type_map = {
            "validation": ActivityType.ERROR_VALIDATION,
            "execution": ActivityType.ERROR_EXECUTION,
            "timeout": ActivityType.ERROR_TIMEOUT,
            "rate_limit": ActivityType.ERROR_RATE_LIMIT,
        }
        return self.stream.emit(
            type_map.get(error_type, ActivityType.ERROR_EXECUTION),
            f"Error: {error_type}",
            message,
            level=ActivityLevel.ERROR,
            data={"error_type": error_type, "message": message},
            **kwargs,
        )


# Global activity stream instance
_activity_stream: Optional[ActivityStream] = None
_activity_emitter: Optional[ActivityEmitter] = None


def get_activity_stream() -> ActivityStream:
    """Get or create the global activity stream."""
    global _activity_stream
    if _activity_stream is None:
        _activity_stream = ActivityStream()
    return _activity_stream


def get_activity_emitter() -> ActivityEmitter:
    """Get or create the global activity emitter."""
    global _activity_emitter
    if _activity_emitter is None:
        _activity_emitter = ActivityEmitter(get_activity_stream())
    return _activity_emitter
