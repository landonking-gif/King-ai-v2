"""
Approval System Models.

Supports multi-approver workflows for high-risk decisions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, List


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    PARTIAL = "partial"  # Some approvals received, more needed
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
class ApprovalVote:
    """A single approval vote from an approver."""
    user_id: str
    decision: str  # "approve" or "reject"
    timestamp: datetime
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.decision not in ("approve", "reject"):
            raise ValueError(f"Invalid decision: {self.decision}")


@dataclass
class ApprovalRequest:
    """A request for human approval with multi-approver support."""
    id: str
    business_id: str
    action_type: ApprovalType
    title: str
    description: str
    risk_level: RiskLevel
    risk_factors: List[RiskFactor] = field(default_factory=list)
    payload: dict = field(default_factory=dict)  # Action details
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Multi-approver support
    required_approvers: int = 1
    votes: List[ApprovalVote] = field(default_factory=list)
    
    # Timestamps and metadata
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
        if self.status in (ApprovalStatus.PENDING, ApprovalStatus.PARTIAL):
            return (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return 0
    
    @property
    def approval_count(self) -> int:
        """Count of approval votes."""
        return sum(1 for v in self.votes if v.decision == "approve")
    
    @property
    def rejection_count(self) -> int:
        """Count of rejection votes."""
        return sum(1 for v in self.votes if v.decision == "reject")
    
    @property
    def is_fully_approved(self) -> bool:
        """Check if enough approvals have been received."""
        return self.approval_count >= self.required_approvers
    
    @property
    def remaining_approvals(self) -> int:
        """Number of additional approvals needed."""
        return max(0, self.required_approvers - self.approval_count)
    
    def has_voted(self, user_id: str) -> bool:
        """Check if a user has already voted."""
        return any(v.user_id == user_id for v in self.votes)
    
    def add_vote(self, user_id: str, decision: str, notes: Optional[str] = None) -> bool:
        """
        Add a vote to the request.
        
        Args:
            user_id: ID of the approver
            decision: "approve" or "reject"
            notes: Optional notes
            
        Returns:
            True if this vote completed the approval process
        """
        # Check if user already voted
        if self.has_voted(user_id):
            return False
        
        # Add the vote
        vote = ApprovalVote(
            user_id=user_id,
            decision=decision,
            timestamp=datetime.utcnow(),
            notes=notes
        )
        self.votes.append(vote)
        
        # Handle rejection - any rejection rejects the whole request
        if decision == "reject":
            self.status = ApprovalStatus.REJECTED
            self.reviewed_at = datetime.utcnow()
            self.reviewed_by = user_id
            self.review_notes = notes
            return False
        
        # Check if fully approved
        if self.is_fully_approved:
            self.status = ApprovalStatus.APPROVED
            self.reviewed_at = datetime.utcnow()
            self.reviewed_by = user_id
            return True
        
        # Partial approval - need more votes
        self.status = ApprovalStatus.PARTIAL
        return False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "business_id": self.business_id,
            "action_type": self.action_type.value,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "required_approvers": self.required_approvers,
            "approval_count": self.approval_count,
            "remaining_approvals": self.remaining_approvals,
            "votes": [
                {
                    "user_id": v.user_id,
                    "decision": v.decision,
                    "timestamp": v.timestamp.isoformat(),
                    "notes": v.notes
                }
                for v in self.votes
            ],
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "waiting_hours": round(self.waiting_hours, 2),
            "is_expired": self.is_expired
        }


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
    action_types: List[ApprovalType]
    min_risk_level: RiskLevel
    auto_approve_below: Optional[float] = None  # Auto-approve if $ below
    require_two_approvers: bool = False
    expiry_hours: int = 24
    expiry_hours: int = 24
    notify_on_create: bool = True
    escalate_after_hours: int = 4
