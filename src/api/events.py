"""
Event Broadcasting System.
"""
from datetime import datetime
from enum import Enum
from typing import Any
from src.api.websocket import manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be broadcast."""
    # Business events
    BUSINESS_CREATED = "business.created"
    BUSINESS_UPDATED = "business.updated"
    BUSINESS_STAGE_CHANGED = "business.stage_changed"
    
    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # Playbook events
    PLAYBOOK_STARTED = "playbook.started"
    PLAYBOOK_COMPLETED = "playbook.completed"
    PLAYBOOK_FAILED = "playbook.failed"
    
    # Approval events
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    
    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC = "system.metric"
    
    # Analytics events
    KPI_UPDATE = "analytics.kpi_update"
    ANOMALY_DETECTED = "analytics.anomaly"


class EventBroadcaster:
    """Broadcast events to WebSocket clients."""

    async def emit(
        self,
        event_type: EventType,
        data: dict,
        business_id: str = None,
        user_id: str = None,
    ):
        """Emit an event to relevant subscribers."""
        event = {
            "type": "event",
            "event": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to specific channels
        if business_id:
            await manager.broadcast_to_channel(f"business:{business_id}", event)
        
        if user_id:
            await manager.broadcast_to_channel(f"user:{user_id}", event)

        # Always broadcast to global for system events
        if event_type.value.startswith("system."):
            await manager.broadcast_to_channel("global", event)

        logger.debug(f"Emitted event: {event_type.value}")

    async def emit_business_update(self, business_id: str, changes: dict):
        """Emit a business update event."""
        await self.emit(
            EventType.BUSINESS_UPDATED,
            {"business_id": business_id, "changes": changes},
            business_id=business_id,
        )

    async def emit_task_progress(
        self,
        business_id: str,
        task_id: str,
        status: str,
        progress: float = None,
        result: Any = None,
    ):
        """Emit task progress event."""
        event_type = {
            "started": EventType.TASK_STARTED,
            "completed": EventType.TASK_COMPLETED,
            "failed": EventType.TASK_FAILED,
        }.get(status, EventType.TASK_STARTED)

        await self.emit(
            event_type,
            {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "result": result,
            },
            business_id=business_id,
        )

    async def emit_approval_required(
        self,
        business_id: str,
        request_id: str,
        title: str,
        risk_level: str,
    ):
        """Emit approval required event."""
        await self.emit(
            EventType.APPROVAL_REQUIRED,
            {
                "request_id": request_id,
                "title": title,
                "risk_level": risk_level,
            },
            business_id=business_id,
        )

    async def emit_kpi_update(self, business_id: str, kpis: dict):
        """Emit KPI update event."""
        await self.emit(
            EventType.KPI_UPDATE,
            {"kpis": kpis},
            business_id=business_id,
        )

    async def emit_system_alert(self, level: str, message: str, details: dict = None):
        """Emit system alert."""
        await self.emit(
            EventType.SYSTEM_ALERT,
            {"level": level, "message": message, "details": details or {}},
        )


# Global broadcaster instance
broadcaster = EventBroadcaster()
