"""
Evolution Models - Data models for evolution proposals and status tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class ProposalStatus(str, Enum):
    """Status of an evolution proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass
class EvolutionProposal:
    """
    Represents an evolution proposal.
    Compatible with database EvolutionProposal model.
    """
    id: str
    type: str
    description: str
    rationale: str
    proposed_changes: Dict[str, Any]
    expected_impact: Optional[str] = None
    confidence_score: float = 0.0
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
