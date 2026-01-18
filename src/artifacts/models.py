"""
Typed Artifact Models.

Provides structured, validated data containers passed between agents
with full provenance tracking for compliance and debugging.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from uuid import uuid4
import hashlib
import json

from pydantic import BaseModel, Field, field_validator


class ArtifactType(str, Enum):
    """Type of artifact being stored."""
    RESEARCH = "research"
    CODE = "code"
    CONTENT = "content"
    BUSINESS_PLAN = "business_plan"
    FINANCE = "finance"
    LEGAL = "legal"
    ANALYSIS = "analysis"
    TASK_RESULT = "task_result"
    GENERIC = "generic"


class SafetyClass(str, Enum):
    """Artifact safety classification for access control."""
    PUBLIC = "public"           # Can be shared externally
    INTERNAL = "internal"       # Internal use only
    SENSITIVE = "sensitive"     # Requires extra care
    PII = "pii"                 # Contains personally identifiable information
    CONFIDENTIAL = "confidential"  # Business confidential


class ProvenanceRecord(BaseModel):
    """Records the lineage of an artifact - who/what created it and from what."""
    
    actor_id: str = Field(description="ID of agent/service that created this")
    actor_type: str = Field(description="Type: agent, service, user, system")
    action: str = Field(description="What action produced this artifact")
    inputs_hash: str = Field(default="", description="Hash of inputs used")
    tool_ids: List[str] = Field(default_factory=list, description="Tools used")
    model_used: Optional[str] = Field(default=None, description="LLM model used if any")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    duration_ms: int = Field(default=0, description="Execution time in ms")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "action": self.action,
            "inputs_hash": self.inputs_hash,
            "tool_ids": self.tool_ids,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class Artifact(BaseModel):
    """
    Base artifact model for all typed artifacts.
    
    Artifacts are the primary data containers passed between agents.
    Each artifact tracks its full provenance for auditability.
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: ArtifactType = Field(default=ArtifactType.GENERIC)
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What this artifact contains")
    content: Dict[str, Any] = Field(default_factory=dict, description="Artifact payload")
    
    # Classification
    safety_class: SafetyClass = Field(default=SafetyClass.INTERNAL)
    tags: List[str] = Field(default_factory=list)
    
    # Provenance
    created_by: str = Field(description="Agent/service that created this")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_artifact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of artifacts this was derived from"
    )
    provenance: Optional[ProvenanceRecord] = None
    
    # Business context
    business_id: Optional[str] = Field(default=None, description="Associated business")
    session_id: Optional[str] = Field(default=None, description="Session that created this")
    
    # Versioning
    version: int = Field(default=1)
    previous_version_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str = Field(default="")
    
    def model_post_init(self, __context: Any) -> None:
        """Compute content hash after initialization."""
        if not self.content_hash and self.content:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of content for integrity verification."""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure content is provided."""
        if v is None:
            return {}
        return v
    
    def derive(
        self,
        name: str,
        content: Dict[str, Any],
        created_by: str,
        artifact_type: Optional[ArtifactType] = None,
    ) -> "Artifact":
        """
        Create a new artifact derived from this one.
        Automatically tracks parent lineage.
        """
        return Artifact(
            name=name,
            artifact_type=artifact_type or self.artifact_type,
            content=content,
            created_by=created_by,
            parent_artifact_ids=[self.id] + self.parent_artifact_ids[:4],  # Keep 5 levels
            business_id=self.business_id,
            session_id=self.session_id,
            safety_class=self.safety_class,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "safety_class": self.safety_class.value,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "parent_artifact_ids": self.parent_artifact_ids,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "business_id": self.business_id,
            "session_id": self.session_id,
            "version": self.version,
            "previous_version_id": self.previous_version_id,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create artifact from dictionary."""
        data = data.copy()
        if "artifact_type" in data and isinstance(data["artifact_type"], str):
            data["artifact_type"] = ArtifactType(data["artifact_type"])
        if "safety_class" in data and isinstance(data["safety_class"], str):
            data["safety_class"] = SafetyClass(data["safety_class"])
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "provenance" in data and data["provenance"]:
            data["provenance"] = ProvenanceRecord(**data["provenance"])
        return cls(**data)


class ResearchArtifact(Artifact):
    """Artifact containing research findings."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.RESEARCH)
    
    # Research-specific fields in content
    # content = {
    #     "summary": str,
    #     "findings": List[str],
    #     "sources": List[Dict],  # {url, title, snippet}
    #     "confidence": float,
    #     "methodology": str,
    # }


class CodeArtifact(Artifact):
    """Artifact containing generated code."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.CODE)
    
    # Code-specific fields in content
    # content = {
    #     "language": str,
    #     "code": str,
    #     "file_path": str,
    #     "tests": List[str],
    #     "dependencies": List[str],
    # }


class ContentArtifact(Artifact):
    """Artifact containing generated content (marketing, docs, etc)."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.CONTENT)
    
    # Content-specific fields
    # content = {
    #     "content_type": str,  # blog, marketing, documentation
    #     "title": str,
    #     "body": str,
    #     "format": str,  # markdown, html, plain
    #     "target_audience": str,
    # }


class BusinessPlanArtifact(Artifact):
    """Artifact containing a business plan."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.BUSINESS_PLAN)
    
    # Business plan fields
    # content = {
    #     "executive_summary": str,
    #     "market_analysis": Dict,
    #     "business_model": Dict,
    #     "financial_projections": Dict,
    #     "marketing_strategy": Dict,
    #     "operations_plan": Dict,
    #     "risk_assessment": Dict,
    # }


class FinanceArtifact(Artifact):
    """Artifact containing financial data/projections."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.FINANCE)
    safety_class: SafetyClass = Field(default=SafetyClass.SENSITIVE)
    
    # Finance fields
    # content = {
    #     "report_type": str,  # projection, statement, analysis
    #     "period": str,
    #     "revenue": Dict,
    #     "expenses": Dict,
    #     "projections": List[Dict],
    # }


class LegalArtifact(Artifact):
    """Artifact containing legal documents."""
    
    artifact_type: ArtifactType = Field(default=ArtifactType.LEGAL)
    safety_class: SafetyClass = Field(default=SafetyClass.CONFIDENTIAL)
    
    # Legal fields
    # content = {
    #     "document_type": str,  # contract, terms, policy
    #     "title": str,
    #     "body": str,
    #     "jurisdiction": str,
    #     "requires_review": bool,
    # }
