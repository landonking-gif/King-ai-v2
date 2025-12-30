"""
Approval System Models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    MODIFIED = "modified"


class RiskLevel(Enum):
    """Risk level requiring approval."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalType(Enum):
    """Types of actions requiring approval."""
    FINANCIAL = "financial"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    TECHNICAL = "technical"
    EXTERNAL = "external"


@dataclass
class RiskFactor:
    """A single risk factor."""
    category: str
    description: str
    severity: RiskLevel
    mitigation: Optional[str] = None


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    business_id: str
    action_type: ApprovalType
    title: str
    description: str
    risk_level: RiskLevel
    risk_factors: list[RiskFactor] = field(default_factory=list)
    payload: dict = field(default_factory=dict)  # Action details
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    modified_payload: Optional[dict] = None
    source_plan_id: Optional[str] = None
    source_task_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at and self.status == ApprovalStatus.PENDING:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def waiting_hours(self) -> float:
        if self.status == ApprovalStatus.PENDING:
            return (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return 0


@dataclass
class ApprovalDecision:
    """Record of an approval decision."""
    request_id: str
    decision: ApprovalStatus
    decided_by: str
    decided_at: datetime
    notes: Optional[str] = None
    modifications: Optional[dict] = None


@dataclass
class ApprovalPolicy:
    """Policy defining when approval is required."""
    id: str
    name: str
    action_types: list[ApprovalType]
    min_risk_level: RiskLevel
    auto_approve_below: Optional[float] = None  # Auto-approve if $ below
    require_two_approvers: bool = False
    expiry_hours: int = 24
    notify_on_create: bool = True
    escalate_after_hours: int = 4
