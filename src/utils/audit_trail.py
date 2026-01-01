"""
Audit Trail Export Functionality.

Provides comprehensive audit trail management for compliance,
reporting, and forensic analysis.
"""

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict
from uuid import uuid4

from src.database.connection import get_db_session
from src.utils.structured_logging import get_logger

logger = get_logger("audit_trail")


class AuditEventType(str, Enum):
    """Types of auditable events."""
    # Business events
    BUSINESS_CREATED = "business.created"
    BUSINESS_UPDATED = "business.updated"
    BUSINESS_DELETED = "business.deleted"
    BUSINESS_STATUS_CHANGED = "business.status_changed"
    
    # Financial events
    TRANSACTION_CREATED = "transaction.created"
    PAYMENT_PROCESSED = "payment.processed"
    REFUND_ISSUED = "refund.issued"
    PAYOUT_COMPLETED = "payout.completed"
    
    # Approval events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_APPROVED = "approval.approved"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_DELEGATED = "approval.delegated"
    
    # System events
    EVOLUTION_PROPOSED = "evolution.proposed"
    EVOLUTION_EXECUTED = "evolution.executed"
    EVOLUTION_ROLLED_BACK = "evolution.rolled_back"
    
    # User events
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_PERMISSION_CHANGED = "user.permission_changed"
    
    # API events
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"
    RATE_LIMIT_EXCEEDED = "api.rate_limit_exceeded"
    
    # Data events
    DATA_EXPORTED = "data.exported"
    DATA_IMPORTED = "data.imported"
    BACKUP_CREATED = "backup.created"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType = AuditEventType.BUSINESS_UPDATED
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor_id: Optional[str] = None
    actor_type: str = "system"  # user, system, agent
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    business_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "severity": self.severity.value if isinstance(self.severity, Enum) else self.severity,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "business_id": self.business_id,
        }


class AuditExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines for streaming


@dataclass
class AuditExportOptions:
    """Options for audit export."""
    format: AuditExportFormat = AuditExportFormat.JSON
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    actor_ids: Optional[List[str]] = None
    business_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    include_metadata: bool = True
    max_records: int = 10000


class AuditTrailManager:
    """
    Manages audit trail recording and export.
    
    Provides:
    - Event recording with automatic enrichment
    - Flexible querying and filtering
    - Multiple export formats
    - Compliance reporting
    """
    
    def __init__(self):
        self._buffer: List[AuditEvent] = []
        self._buffer_size = 100
        self._flush_interval = 60  # seconds
    
    async def record(
        self,
        event_type: AuditEventType,
        resource_type: str,
        resource_id: str,
        action: str,
        actor_id: Optional[str] = None,
        actor_type: str = "system",
        old_value: Optional[Dict] = None,
        new_value: Optional[Dict] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict] = None,
        request_context: Optional[Dict] = None,
        business_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Record an audit event.
        
        Args:
            event_type: Type of event
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            action: Description of action taken
            actor_id: ID of actor (user/agent) performing action
            actor_type: Type of actor (user, system, agent)
            old_value: Previous state of resource
            new_value: New state of resource
            severity: Severity level
            metadata: Additional context
            request_context: HTTP request context (IP, user agent, etc.)
            business_id: Related business ID
        
        Returns:
            The recorded audit event
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
            business_id=business_id,
        )
        
        # Enrich from request context
        if request_context:
            event.ip_address = request_context.get("ip_address")
            event.user_agent = request_context.get("user_agent")
            event.request_id = request_context.get("request_id")
        
        # Buffer for batch writing
        self._buffer.append(event)
        
        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            await self._flush_buffer()
        
        logger.info(
            "Audit event recorded",
            event_type=event.event_type.value,
            resource_id=resource_id,
            actor_id=actor_id
        )
        
        return event
    
    async def _flush_buffer(self):
        """Flush buffered events to database."""
        if not self._buffer:
            return
        
        events = self._buffer.copy()
        self._buffer.clear()
        
        try:
            async with get_db_session() as session:
                from sqlalchemy import text
                
                for event in events:
                    await session.execute(
                        text("""
                            INSERT INTO audit_trail 
                            (id, event_type, severity, timestamp, actor_id, actor_type,
                             resource_type, resource_id, action, old_value, new_value,
                             metadata, ip_address, user_agent, request_id, business_id)
                            VALUES 
                            (:id, :event_type, :severity, :timestamp, :actor_id, :actor_type,
                             :resource_type, :resource_id, :action, :old_value, :new_value,
                             :metadata, :ip_address, :user_agent, :request_id, :business_id)
                        """),
                        {
                            "id": event.id,
                            "event_type": event.event_type.value,
                            "severity": event.severity.value,
                            "timestamp": event.timestamp,
                            "actor_id": event.actor_id,
                            "actor_type": event.actor_type,
                            "resource_type": event.resource_type,
                            "resource_id": event.resource_id,
                            "action": event.action,
                            "old_value": json.dumps(event.old_value) if event.old_value else None,
                            "new_value": json.dumps(event.new_value) if event.new_value else None,
                            "metadata": json.dumps(event.metadata) if event.metadata else None,
                            "ip_address": event.ip_address,
                            "user_agent": event.user_agent,
                            "request_id": event.request_id,
                            "business_id": event.business_id,
                        }
                    )
                await session.commit()
                
        except Exception as e:
            logger.error("Failed to flush audit buffer", error=str(e))
            # Re-add events to buffer for retry
            self._buffer.extend(events)
    
    async def query(
        self,
        options: AuditExportOptions,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.
        
        Args:
            options: Query and filter options
        
        Returns:
            List of matching audit events
        """
        try:
            async with get_db_session() as session:
                from sqlalchemy import text
                
                # Build query with filters
                conditions = ["1=1"]
                params = {"limit": options.max_records}
                
                if options.start_date:
                    conditions.append("timestamp >= :start_date")
                    params["start_date"] = options.start_date
                
                if options.end_date:
                    conditions.append("timestamp <= :end_date")
                    params["end_date"] = options.end_date
                
                if options.event_types:
                    event_type_list = [et.value for et in options.event_types]
                    conditions.append(f"event_type IN ({','.join([':et_' + str(i) for i in range(len(event_type_list))])})")
                    for i, et in enumerate(event_type_list):
                        params[f"et_{i}"] = et
                
                if options.severities:
                    severity_list = [s.value for s in options.severities]
                    conditions.append(f"severity IN ({','.join([':sev_' + str(i) for i in range(len(severity_list))])})")
                    for i, s in enumerate(severity_list):
                        params[f"sev_{i}"] = s
                
                if options.business_ids:
                    conditions.append(f"business_id IN ({','.join([':bid_' + str(i) for i in range(len(options.business_ids))])})")
                    for i, bid in enumerate(options.business_ids):
                        params[f"bid_{i}"] = bid
                
                query = f"""
                    SELECT id, event_type, severity, timestamp, actor_id, actor_type,
                           resource_type, resource_id, action, old_value, new_value,
                           metadata, ip_address, user_agent, request_id, business_id
                    FROM audit_trail
                    WHERE {' AND '.join(conditions)}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                events = []
                for row in rows:
                    events.append(AuditEvent(
                        id=row[0],
                        event_type=AuditEventType(row[1]) if row[1] in [e.value for e in AuditEventType] else row[1],
                        severity=AuditSeverity(row[2]) if row[2] in [s.value for s in AuditSeverity] else row[2],
                        timestamp=row[3],
                        actor_id=row[4],
                        actor_type=row[5],
                        resource_type=row[6],
                        resource_id=row[7],
                        action=row[8],
                        old_value=json.loads(row[9]) if row[9] else None,
                        new_value=json.loads(row[10]) if row[10] else None,
                        metadata=json.loads(row[11]) if row[11] else {},
                        ip_address=row[12],
                        user_agent=row[13],
                        request_id=row[14],
                        business_id=row[15],
                    ))
                
                return events
                
        except Exception as e:
            logger.error("Failed to query audit trail", error=str(e))
            return []
    
    async def export(
        self,
        options: AuditExportOptions,
    ) -> tuple[str, str]:
        """
        Export audit trail in specified format.
        
        Args:
            options: Export options including format and filters
        
        Returns:
            Tuple of (content, content_type)
        """
        events = await self.query(options)
        
        if options.format == AuditExportFormat.JSON:
            content = json.dumps(
                [e.to_dict() for e in events],
                indent=2,
                default=str
            )
            return content, "application/json"
        
        elif options.format == AuditExportFormat.JSONL:
            lines = [json.dumps(e.to_dict(), default=str) for e in events]
            content = "\n".join(lines)
            return content, "application/x-jsonlines"
        
        elif options.format == AuditExportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            headers = [
                "id", "event_type", "severity", "timestamp", 
                "actor_id", "actor_type", "resource_type", "resource_id",
                "action", "business_id", "ip_address"
            ]
            if options.include_metadata:
                headers.extend(["old_value", "new_value", "metadata"])
            writer.writerow(headers)
            
            # Data rows
            for event in events:
                row = [
                    event.id,
                    event.event_type.value if isinstance(event.event_type, Enum) else event.event_type,
                    event.severity.value if isinstance(event.severity, Enum) else event.severity,
                    event.timestamp.isoformat() if event.timestamp else "",
                    event.actor_id or "",
                    event.actor_type,
                    event.resource_type,
                    event.resource_id,
                    event.action,
                    event.business_id or "",
                    event.ip_address or "",
                ]
                if options.include_metadata:
                    row.extend([
                        json.dumps(event.old_value) if event.old_value else "",
                        json.dumps(event.new_value) if event.new_value else "",
                        json.dumps(event.metadata) if event.metadata else "",
                    ])
                writer.writerow(row)
            
            content = output.getvalue()
            return content, "text/csv"
        
        raise ValueError(f"Unsupported export format: {options.format}")
    
    async def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        business_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for the given period.
        
        Returns summary statistics and notable events.
        """
        options = AuditExportOptions(
            start_date=start_date,
            end_date=end_date,
            business_ids=[business_id] if business_id else None,
            max_records=50000,
        )
        
        events = await self.query(options)
        
        # Aggregate statistics
        event_counts = {}
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        actor_counts = {}
        daily_counts = {}
        
        for event in events:
            # By event type
            et = event.event_type.value if isinstance(event.event_type, Enum) else str(event.event_type)
            event_counts[et] = event_counts.get(et, 0) + 1
            
            # By severity
            sev = event.severity.value if isinstance(event.severity, Enum) else str(event.severity)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            # By actor
            if event.actor_id:
                actor_counts[event.actor_id] = actor_counts.get(event.actor_id, 0) + 1
            
            # By day
            if event.timestamp:
                day = event.timestamp.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
        
        # Find notable events (critical severity)
        critical_events = [
            e.to_dict() for e in events 
            if e.severity == AuditSeverity.CRITICAL
        ][:50]
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "by_type": event_counts,
                "by_severity": severity_counts,
                "unique_actors": len(actor_counts),
                "daily_average": len(events) / max(len(daily_counts), 1),
            },
            "top_actors": sorted(
                actor_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "daily_trend": [
                {"date": k, "count": v}
                for k, v in sorted(daily_counts.items())
            ],
            "critical_events": critical_events,
            "generated_at": datetime.utcnow().isoformat(),
        }


# Singleton instance
audit_manager = AuditTrailManager()


# Convenience functions for common operations
async def audit_business_created(business_id: str, business_data: Dict, actor_id: str = None):
    """Record business creation event."""
    await audit_manager.record(
        event_type=AuditEventType.BUSINESS_CREATED,
        resource_type="business",
        resource_id=business_id,
        action="Created new business unit",
        actor_id=actor_id,
        new_value=business_data,
        severity=AuditSeverity.INFO,
        business_id=business_id,
    )


async def audit_approval_action(
    approval_id: str,
    action: str,
    actor_id: str,
    old_status: str = None,
    new_status: str = None,
    business_id: str = None,
):
    """Record approval action event."""
    event_type = {
        "approved": AuditEventType.APPROVAL_APPROVED,
        "rejected": AuditEventType.APPROVAL_REJECTED,
        "delegated": AuditEventType.APPROVAL_DELEGATED,
    }.get(action, AuditEventType.APPROVAL_REQUESTED)
    
    await audit_manager.record(
        event_type=event_type,
        resource_type="approval",
        resource_id=approval_id,
        action=f"Approval {action}",
        actor_id=actor_id,
        old_value={"status": old_status} if old_status else None,
        new_value={"status": new_status} if new_status else None,
        severity=AuditSeverity.INFO,
        business_id=business_id,
    )


async def audit_evolution_event(
    proposal_id: str,
    action: str,
    proposal_data: Dict = None,
    success: bool = True,
):
    """Record evolution/self-modification event."""
    event_type = {
        "proposed": AuditEventType.EVOLUTION_PROPOSED,
        "executed": AuditEventType.EVOLUTION_EXECUTED,
        "rolled_back": AuditEventType.EVOLUTION_ROLLED_BACK,
    }.get(action, AuditEventType.EVOLUTION_PROPOSED)
    
    await audit_manager.record(
        event_type=event_type,
        resource_type="evolution",
        resource_id=proposal_id,
        action=f"Evolution {action}",
        actor_type="system",
        new_value=proposal_data,
        severity=AuditSeverity.CRITICAL if not success else AuditSeverity.WARNING,
        metadata={"success": success},
    )
