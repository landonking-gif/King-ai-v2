"""
Evolution Models - Data models for evolution proposals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class ProposalStatus(str, Enum):
    """Status of an evolution proposal."""
    PENDING = "pending"
    TESTING = "testing"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass
class EvolutionProposal:
    """An evolution proposal for code improvements."""
    id: str
    title: str
    description: str
    file_paths: List[str]
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    impact_score: float = 0.0
    risk_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "file_paths": self.file_paths,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "impact_score": self.impact_score,
            "risk_level": self.risk_level,
            "metadata": self.metadata
        }
