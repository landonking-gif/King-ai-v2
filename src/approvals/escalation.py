"""
Approval Escalation Service.
Handles automatic escalation of stale approvals with notification chains.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Coroutine
from enum import Enum

from src.approvals.models import (
    ApprovalRequest, ApprovalStatus, RiskLevel
)
from src.utils.structured_logging import get_logger

logger = get_logger("escalation")


class EscalationLevel(str, Enum):
    """Escalation levels."""
    NONE = "none"
    FIRST = "first"  # First reminder
    SECOND = "second"  # Escalate to manager
    URGENT = "urgent"  # Urgent escalation
    CRITICAL = "critical"  # Executive escalation


@dataclass
class EscalationRule:
    """Rule for when to escalate."""
    level: EscalationLevel
    after_minutes: int
    notify_roles: List[str]
    message_template: str
    require_acknowledgment: bool = False


@dataclass
class EscalationPolicy:
    """Policy defining escalation behavior."""
    name: str
    min_risk_level: RiskLevel
    rules: List[EscalationRule]
    sla_minutes: int = 60  # SLA for approval
    enabled: bool = True


@dataclass
class EscalationEvent:
    """Record of an escalation event."""
    request_id: str
    level: EscalationLevel
    escalated_at: datetime
    escalated_to: List[str]
    message: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# Default escalation policies
DEFAULT_POLICIES = [
    EscalationPolicy(
        name="critical_financial",
        min_risk_level=RiskLevel.CRITICAL,
        sla_minutes=30,
        rules=[
            EscalationRule(
                level=EscalationLevel.FIRST,
                after_minutes=15,
                notify_roles=["approver"],
                message_template="URGENT: Critical approval pending for {title}. Awaiting response for {wait_time}.",
            ),
            EscalationRule(
                level=EscalationLevel.SECOND,
                after_minutes=25,
                notify_roles=["approver", "manager"],
                message_template="ESCALATION: Critical approval for {title} has exceeded {wait_time}. Manager attention required.",
            ),
            EscalationRule(
                level=EscalationLevel.CRITICAL,
                after_minutes=30,
                notify_roles=["approver", "manager", "executive"],
                message_template="CRITICAL: SLA breach for {title}. Executive intervention required.",
                require_acknowledgment=True,
            ),
        ],
    ),
    EscalationPolicy(
        name="high_risk",
        min_risk_level=RiskLevel.HIGH,
        sla_minutes=120,
        rules=[
            EscalationRule(
                level=EscalationLevel.FIRST,
                after_minutes=60,
                notify_roles=["approver"],
                message_template="Reminder: High-risk approval pending for {title}. Waiting for {wait_time}.",
            ),
            EscalationRule(
                level=EscalationLevel.SECOND,
                after_minutes=90,
                notify_roles=["approver", "manager"],
                message_template="Escalation: High-risk approval for {title} pending for {wait_time}.",
            ),
            EscalationRule(
                level=EscalationLevel.URGENT,
                after_minutes=120,
                notify_roles=["manager"],
                message_template="SLA Warning: High-risk approval {title} approaching SLA breach.",
                require_acknowledgment=True,
            ),
        ],
    ),
    EscalationPolicy(
        name="standard",
        min_risk_level=RiskLevel.LOW,
        sla_minutes=1440,  # 24 hours
        rules=[
            EscalationRule(
                level=EscalationLevel.FIRST,
                after_minutes=240,  # 4 hours
                notify_roles=["approver"],
                message_template="Reminder: Approval pending for {title}.",
            ),
            EscalationRule(
                level=EscalationLevel.SECOND,
                after_minutes=720,  # 12 hours
                notify_roles=["approver", "manager"],
                message_template="Escalation: Approval for {title} pending for {wait_time}.",
            ),
        ],
    ),
]


class EscalationService:
    """
    Service for managing approval escalations.
    
    Features:
    - Automatic escalation based on wait time
    - Configurable escalation policies
    - Notification chains
    - SLA tracking
    - Acknowledgment handling
    """
    
    def __init__(self):
        self._policies: Dict[str, EscalationPolicy] = {
            p.name: p for p in DEFAULT_POLICIES
        }
        self._escalation_history: Dict[str, List[EscalationEvent]] = {}
        self._notification_callback: Optional[Callable] = None
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        self._check_interval = 60  # Check every minute
    
    def register_notification_callback(
        self,
        callback: Callable[[EscalationEvent, ApprovalRequest], Coroutine[Any, Any, None]],
    ) -> None:
        """Register callback for sending escalation notifications."""
        self._notification_callback = callback
    
    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add a custom escalation policy."""
        self._policies[policy.name] = policy
    
    def get_policy_for_request(
        self, request: ApprovalRequest
    ) -> Optional[EscalationPolicy]:
        """Find the applicable policy for a request."""
        applicable = [
            p for p in self._policies.values()
            if p.enabled and request.risk_level.value >= p.min_risk_level.value
        ]
        
        if not applicable:
            return None
        
        # Return the most restrictive (highest min risk level)
        return max(applicable, key=lambda p: p.min_risk_level.value)
    
    def get_current_escalation_level(
        self, request: ApprovalRequest
    ) -> EscalationLevel:
        """Get the current escalation level for a request."""
        history = self._escalation_history.get(request.id, [])
        if not history:
            return EscalationLevel.NONE
        return history[-1].level
    
    def get_escalation_history(
        self, request_id: str
    ) -> List[EscalationEvent]:
        """Get escalation history for a request."""
        return self._escalation_history.get(request_id, [])
    
    async def check_and_escalate(
        self, request: ApprovalRequest
    ) -> Optional[EscalationEvent]:
        """
        Check if a request needs escalation and escalate if needed.
        
        Returns:
            EscalationEvent if escalated, None otherwise
        """
        if request.status != ApprovalStatus.PENDING:
            return None
        
        policy = self.get_policy_for_request(request)
        if not policy:
            return None
        
        # Calculate wait time
        wait_minutes = (
            datetime.utcnow() - request.created_at
        ).total_seconds() / 60
        
        # Get current escalation level
        current_level = self.get_current_escalation_level(request)
        current_level_index = list(EscalationLevel).index(current_level)
        
        # Find the next applicable rule
        for rule in policy.rules:
            rule_level_index = list(EscalationLevel).index(rule.level)
            
            # Skip rules we've already passed
            if rule_level_index <= current_level_index:
                continue
            
            # Check if it's time to escalate
            if wait_minutes >= rule.after_minutes:
                return await self._escalate(request, rule, wait_minutes)
        
        return None
    
    async def _escalate(
        self,
        request: ApprovalRequest,
        rule: EscalationRule,
        wait_minutes: float,
    ) -> EscalationEvent:
        """Perform escalation."""
        # Format wait time
        if wait_minutes < 60:
            wait_time = f"{int(wait_minutes)} minutes"
        elif wait_minutes < 1440:
            wait_time = f"{wait_minutes / 60:.1f} hours"
        else:
            wait_time = f"{wait_minutes / 1440:.1f} days"
        
        # Create message
        message = rule.message_template.format(
            title=request.title,
            wait_time=wait_time,
            risk_level=request.risk_level.value,
            business_id=request.business_id,
        )
        
        # Create escalation event
        event = EscalationEvent(
            request_id=request.id,
            level=rule.level,
            escalated_at=datetime.utcnow(),
            escalated_to=rule.notify_roles,
            message=message,
        )
        
        # Record in history
        if request.id not in self._escalation_history:
            self._escalation_history[request.id] = []
        self._escalation_history[request.id].append(event)
        
        # Send notification
        if self._notification_callback:
            try:
                await self._notification_callback(event, request)
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")
        
        logger.info(
            f"Escalated request {request.id} to level {rule.level.value}",
            request_id=request.id,
            level=rule.level.value,
            escalated_to=rule.notify_roles,
        )
        
        return event
    
    async def acknowledge_escalation(
        self,
        request_id: str,
        user_id: str,
    ) -> bool:
        """Acknowledge an escalation."""
        history = self._escalation_history.get(request_id, [])
        if not history:
            return False
        
        # Acknowledge the most recent escalation
        latest = history[-1]
        if not latest.acknowledged:
            latest.acknowledged = True
            latest.acknowledged_by = user_id
            latest.acknowledged_at = datetime.utcnow()
            
            logger.info(
                f"Escalation acknowledged for {request_id}",
                request_id=request_id,
                acknowledged_by=user_id,
            )
            return True
        
        return False
    
    async def start_monitoring(
        self,
        get_pending_requests: Callable[[], Coroutine[Any, Any, List[ApprovalRequest]]],
    ) -> None:
        """
        Start background monitoring for escalations.
        
        Args:
            get_pending_requests: Async function that returns pending requests
        """
        self._running = True
        
        async def monitor_loop():
            while self._running:
                try:
                    requests = await get_pending_requests()
                    
                    for request in requests:
                        await self.check_and_escalate(request)
                    
                except Exception as e:
                    logger.error(f"Escalation monitoring error: {e}")
                
                await asyncio.sleep(self._check_interval)
        
        self._check_task = asyncio.create_task(monitor_loop())
        logger.info("Escalation monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Escalation monitoring stopped")
    
    def get_sla_status(
        self, request: ApprovalRequest
    ) -> Dict[str, Any]:
        """Get SLA status for a request."""
        policy = self.get_policy_for_request(request)
        if not policy:
            return {"has_sla": False}
        
        wait_minutes = (
            datetime.utcnow() - request.created_at
        ).total_seconds() / 60
        
        sla_remaining = policy.sla_minutes - wait_minutes
        sla_percent = (wait_minutes / policy.sla_minutes) * 100
        
        return {
            "has_sla": True,
            "sla_minutes": policy.sla_minutes,
            "wait_minutes": round(wait_minutes, 1),
            "sla_remaining_minutes": round(max(0, sla_remaining), 1),
            "sla_percent_used": round(min(100, sla_percent), 1),
            "sla_breached": sla_remaining < 0,
            "current_escalation_level": self.get_current_escalation_level(request).value,
            "escalation_count": len(self._escalation_history.get(request.id, [])),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get escalation statistics."""
        total_escalations = sum(
            len(events) for events in self._escalation_history.values()
        )
        
        acknowledged = sum(
            1 for events in self._escalation_history.values()
            for e in events if e.acknowledged
        )
        
        by_level = {}
        for events in self._escalation_history.values():
            for e in events:
                by_level[e.level.value] = by_level.get(e.level.value, 0) + 1
        
        return {
            "total_escalations": total_escalations,
            "acknowledged": acknowledged,
            "pending_acknowledgment": total_escalations - acknowledged,
            "by_level": by_level,
            "requests_escalated": len(self._escalation_history),
            "monitoring_active": self._running,
        }
    
    def clear_history(self, request_id: str) -> None:
        """Clear escalation history for a resolved request."""
        self._escalation_history.pop(request_id, None)


# Global instance
escalation_service = EscalationService()


def get_escalation_service() -> EscalationService:
    """Get the global escalation service instance."""
    return escalation_service
