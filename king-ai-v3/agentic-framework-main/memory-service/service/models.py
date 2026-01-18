"""
Memory Service Data Models.

Module: memory-service/service/models.py
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlmodel import SQLModel, Field as SQLField, Column, JSON


class SafetyClass(str, Enum):
    """Artifact safety classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PII = "pii"


class ArtifactType(str, Enum):
    """Type of artifact being stored."""

    RESEARCH_SNIPPET = "research_snippet"
    CLAIM_VERIFICATION = "claim_verification"
    CODE_PATCH = "code_patch"
    GENERIC = "generic"


class CompactionStrategy(str, Enum):
    """Memory compaction strategy."""

    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    NONE = "none"


# SQLModel for Postgres storage
class ProvenanceLog(SQLModel, table=True):
    """Provenance log entry for tracking artifact lineage."""

    __tablename__ = "provenance_logs"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    artifact_id: str = SQLField(index=True, nullable=False)
    actor_id: str = SQLField(nullable=False)
    actor_type: str = SQLField(nullable=False)
    inputs_hash: str = SQLField(nullable=False)
    outputs_hash: str = SQLField(nullable=False)
    tool_ids: List[str] = SQLField(sa_column=Column(JSON), default_factory=list)
    parent_artifact_ids: List[str] = SQLField(sa_column=Column(JSON), default_factory=list)
    extra_metadata: Dict[str, Any] = SQLField(sa_column=Column(JSON), default_factory=dict)
    created_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)


class ArtifactRecord(SQLModel, table=True):
    """Structured artifact metadata stored in Postgres."""

    __tablename__ = "artifacts"

    id: str = SQLField(primary_key=True, default_factory=lambda: str(uuid4()))
    artifact_type: str = SQLField(index=True, nullable=False)
    content_hash: str = SQLField(index=True, nullable=False)
    safety_class: str = SQLField(default=SafetyClass.INTERNAL.value, nullable=False)
    created_by: str = SQLField(nullable=False)
    created_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)
    extra_metadata: Dict[str, Any] = SQLField(sa_column=Column(JSON), default_factory=dict)
    embedding_ref: Optional[str] = SQLField(default=None)
    cold_storage_ref: Optional[str] = SQLField(default=None)
    session_id: Optional[str] = SQLField(index=True, default=None)
    token_count: Optional[int] = SQLField(default=None)
    tags: List[str] = SQLField(sa_column=Column(JSON), default_factory=list)


# Pydantic models for API requests/responses
class Artifact(BaseModel):
    """Generic artifact model for API operations."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    artifact_type: ArtifactType = Field(default=ArtifactType.GENERIC)
    content: Dict[str, Any] = Field(description="Artifact content as JSON")
    safety_class: SafetyClass = Field(default=SafetyClass.INTERNAL)
    created_by: str = Field(description="Actor ID that created this artifact")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    parent_artifact_ids: List[str] = Field(
        default_factory=list, description="IDs of parent artifacts"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure content is not empty."""
        if not v:
            raise ValueError("Artifact content cannot be empty")
        return v


class ResearchSnippet(BaseModel):
    """Research snippet artifact (from PRD)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: Dict[str, str] = Field(description="Source metadata (url or doc_id)")
    text: str = Field(min_length=1)
    summary: str
    tags: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance_refs: List[str] = Field(default_factory=list)
    embedding_ref: Optional[str] = None
    safety_class: SafetyClass = Field(default=SafetyClass.INTERNAL)
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CommitRequest(BaseModel):
    """Request to commit an artifact to memory."""

    artifact: Artifact
    actor_id: str = Field(description="ID of actor committing artifact")
    actor_type: str = Field(description="Type of actor (subagent, lead_agent, skill)")
    tool_ids: List[str] = Field(default_factory=list, description="Tools used to create artifact")
    generate_embedding: bool = Field(default=True, description="Generate vector embedding")
    store_in_cold: bool = Field(default=False, description="Store in cold storage (S3/MinIO)")


class CommitResponse(BaseModel):
    """Response from committing an artifact."""

    memory_id: str = Field(description="Unique ID for the committed artifact")
    artifact_id: str
    embedding_generated: bool
    cold_storage_ref: Optional[str] = None
    provenance_id: int
    created_at: datetime


class QueryRequest(BaseModel):
    """Request to query similar artifacts."""

    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=100)
    filter_artifact_type: Optional[ArtifactType] = None
    filter_session_id: Optional[str] = None
    filter_safety_class: Optional[SafetyClass] = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("query_text", "query_embedding")
    @classmethod
    def validate_query_input(cls, v: Any, info: Any) -> Any:
        """Ensure at least one query input is provided."""
        return v


class QueryResult(BaseModel):
    """Single query result item."""

    artifact_id: str
    artifact_type: str
    content: Dict[str, Any]
    similarity: float
    metadata: Dict[str, Any]
    created_by: str
    created_at: datetime


class QueryResponse(BaseModel):
    """Response from querying artifacts."""

    items: List[QueryResult]
    query_time_ms: float
    total_candidates: int


class ProvenanceEntry(BaseModel):
    """Single provenance log entry."""

    artifact_id: str
    actor_id: str
    actor_type: str
    inputs_hash: str
    outputs_hash: str
    tool_ids: List[str]
    parent_artifact_ids: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


class ProvenanceChain(BaseModel):
    """Complete provenance chain for an artifact."""

    artifact_id: str
    chain: List[ProvenanceEntry]
    depth: int
    root_artifacts: List[str]


class CompactionRequest(BaseModel):
    """Request to compact memory for token budget."""

    session_id: str
    strategy: CompactionStrategy = Field(default=CompactionStrategy.SUMMARIZE)
    target_tokens: int = Field(default=8000, ge=1000)
    preserve_artifact_ids: List[str] = Field(
        default_factory=list, description="Artifacts to preserve during compaction"
    )


class CompactionResponse(BaseModel):
    """Response from memory compaction."""

    session_id: str
    tokens_before: int
    tokens_after: int
    artifacts_compacted: int
    artifacts_removed: int
    strategy_used: CompactionStrategy
    summary_artifact_id: Optional[str] = None
