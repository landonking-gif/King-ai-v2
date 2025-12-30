"""
Data models for the evolution engine.
Supports proposal generation, validation, and execution tracking.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime
from uuid import uuid4


class ProposalStatus(str, Enum):
    """Status of an evolution proposal."""
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    # Legacy statuses for backward compatibility
    PENDING = "pending"
    TESTING = "testing"
    APPLIED = "applied"


class ProposalType(str, Enum):
    """Type of evolution proposal."""
    CODE_MODIFICATION = "code_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    MODEL_RETRAINING = "model_retraining"
    AGENT_ENHANCEMENT = "agent_enhancement"
    INFRASTRUCTURE_UPDATE = "infrastructure_update"
    BUSINESS_RULE_CHANGE = "business_rule_change"


class RiskLevel(str, Enum):
    """Risk classification for proposals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceScore(BaseModel):
    """Confidence score for a proposal."""
    overall: float = Field(..., ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    calculated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('overall')
    @classmethod
    def validate_overall(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall confidence must be between 0.0 and 1.0')
        return v


class ValidationResult(BaseModel):
    """Result of proposal validation."""
    passed: bool
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    validated_at: datetime = Field(default_factory=datetime.now)


class CodeChange(BaseModel):
    """A single code change in a proposal."""
    file_path: str
    change_type: Literal["add", "modify", "delete", "rename"]
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    description: str = ""
    
    @property
    def is_safe(self) -> bool:
        """Check if this change appears safe."""
        # Simple heuristics for safety
        if self.change_type == "delete":
            return False  # Deletions are risky
        if self.old_content and len(self.old_content) > 1000:
            return False  # Large changes are risky
        return True


class EvolutionProposal(BaseModel):
    """A proposal for system evolution."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    proposal_type: ProposalType
    
    # Status and lifecycle
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Author and approval
    proposed_by: str = "master_ai"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None
    
    # Risk and confidence
    risk_level: RiskLevel = RiskLevel.MEDIUM
    confidence_score: Optional[ConfidenceScore] = None
    requires_approval: bool = True
    
    # Content
    changes: List[CodeChange] = Field(default_factory=list)
    configuration_changes: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation and execution
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[Dict[str, Any]] = None
    rollback_data: Optional[Dict[str, Any]] = None
    
    # Relationships
    parent_proposal_id: Optional[str] = None
    related_proposals: List[str] = Field(default_factory=list)
    
    # Metrics
    estimated_impact: Dict[str, float] = Field(default_factory=dict)  # e.g., {"performance": 0.1, "reliability": -0.05}
    actual_impact: Optional[Dict[str, float]] = None
    
    # Legacy fields for backward compatibility
    file_paths: List[str] = Field(default_factory=list)
    impact_score: float = 0.0
    
    def can_execute(self) -> bool:
        """Check if proposal is ready for execution."""
        return (
            self.status in [ProposalStatus.READY, ProposalStatus.APPROVED] and
            self.confidence_score and
            self.confidence_score.overall >= 0.7 and
            self.validation_result and
            self.validation_result.passed
        )
    
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk proposal."""
        return (
            self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            any(not change.is_safe for change in self.changes) or
            len(self.changes) > 10  # Many changes = high risk
        )
    
    def calculate_risk_level(self) -> RiskLevel:
        """Calculate risk level based on proposal characteristics."""
        risk_factors = 0
        
        # Type-based risk
        if self.proposal_type in [ProposalType.CODE_MODIFICATION, ProposalType.MODEL_RETRAINING]:
            risk_factors += 2
        elif self.proposal_type == ProposalType.INFRASTRUCTURE_UPDATE:
            risk_factors += 3
        
        # Change-based risk
        if len(self.changes) > 5:
            risk_factors += 1
        if any(not change.is_safe for change in self.changes):
            risk_factors += 2
        
        # Confidence-based risk
        if self.confidence_score and self.confidence_score.overall < 0.8:
            risk_factors += 1
        
        # Map to risk level
        if risk_factors >= 4:
            return RiskLevel.CRITICAL
        elif risk_factors >= 3:
            return RiskLevel.HIGH
        elif risk_factors >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "file_paths": self.file_paths or [c.file_path for c in self.changes],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "impact_score": self.impact_score,
            "risk_level": self.risk_level.value,
            "metadata": self.metadata,
            "confidence_score": self.confidence_score.dict() if self.confidence_score else None,
            "validation_result": self.validation_result.dict() if self.validation_result else None
        }


class EvolutionHistory(BaseModel):
    """History of evolution events."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: Literal["proposed", "approved", "executed", "failed", "rolled_back"]
    proposal_id: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None


class EvolutionMetrics(BaseModel):
    """Metrics for evolution performance."""
    total_proposals: int = 0
    successful_proposals: int = 0
    failed_proposals: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    risk_distribution: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_proposals == 0:
            return 0.0
        return self.successful_proposals / self.total_proposals
    
    def update_from_proposal(self, proposal: EvolutionProposal):
        """Update metrics from a completed proposal."""
        self.total_proposals += 1
        
        if proposal.status == ProposalStatus.COMPLETED:
            self.successful_proposals += 1
        elif proposal.status == ProposalStatus.FAILED:
            self.failed_proposals += 1
        
        if proposal.confidence_score:
            # Rolling average
            self.average_confidence = (
                (self.average_confidence * (self.total_proposals - 1)) +
                proposal.confidence_score.overall
            ) / self.total_proposals
        
        # Update risk distribution
        risk = proposal.risk_level.value
        self.risk_distribution[risk] = self.risk_distribution.get(risk, 0) + 1
        
        self.last_updated = datetime.now()
