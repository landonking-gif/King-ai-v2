"""
Provenance Tracking for Enterprise Agent Audit Trails.

Module: subagent-manager/service/provenance.py

Provides immutable provenance records for all agent actions, enabling
full audit trails and compliance verification.
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ActorType(str, Enum):
    """Types of actors that can create provenance records."""

    ENTERPRISE_AGENT = "enterprise_agent"
    REFLECTIVE_AGENT = "reflective_agent"
    SUBAGENT = "subagent"
    TOOL = "tool"
    HUMAN = "human"
    SYSTEM = "system"


class ArtifactType(str, Enum):
    """Types of artifacts tracked by provenance."""

    EXECUTION_PLAN = "execution_plan"
    EXECUTION_RESULT = "execution_result"
    VALIDATION_RESULT = "validation_result"
    FILE_OUTPUT = "file_output"
    CODE_CHANGE = "code_change"
    API_RESPONSE = "api_response"
    LLM_RESPONSE = "llm_response"
    THINKING_TRACE = "thinking_trace"


class ProvenanceRecord(BaseModel):
    """
    Immutable record of an action for audit trail.

    Each record captures who did what, with what inputs, producing what outputs,
    and when - enabling full reconstruction of execution history.
    """

    provenance_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Actor information
    actor_id: str = Field(..., description="ID of the entity performing the action")
    actor_type: ActorType = Field(...)

    # Action details
    action: str = Field(..., description="Description of the action performed")
    action_category: Optional[str] = Field(None, description="Category of action")

    # Input/Output hashes for integrity verification
    inputs_hash: str = Field(..., description="SHA-256 hash of inputs")
    outputs_hash: str = Field(..., description="SHA-256 hash of outputs")

    # Optional tool information
    tool_id: Optional[str] = Field(None, description="Tool used if applicable")
    tool_version: Optional[str] = Field(None)

    # Chain information
    parent_provenance_id: Optional[str] = Field(
        None, description="Parent record for chain tracking"
    )
    execution_id: Optional[str] = Field(
        None, description="ID of overall execution context"
    )

    # Artifact references
    input_artifact_ids: List[str] = Field(default_factory=list)
    output_artifact_ids: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceChain(BaseModel):
    """A chain of provenance records for an execution."""

    chain_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str = Field(...)
    records: List[ProvenanceRecord] = Field(default_factory=list)
    root_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = Field(None)
    is_valid: bool = Field(default=True)


class Artifact(BaseModel):
    """An artifact with provenance tracking."""

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: ArtifactType = Field(...)
    content_hash: str = Field(..., description="SHA-256 hash of content")
    content: Optional[Any] = Field(None, description="Actual content if stored")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="Actor who created this artifact")
    provenance_id: str = Field(..., description="Provenance record for creation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceTracker:
    """
    Tracks provenance for all agent operations.

    Provides:
    - Cryptographic hashing of inputs/outputs
    - Immutable provenance record creation
    - Chain of custody tracking
    - Integrity verification
    """

    def __init__(
        self,
        hash_algorithm: str = "sha256",
        storage_backend: Optional[Any] = None,
    ):
        """
        Initialize provenance tracker.

        Args:
            hash_algorithm: Algorithm for hashing (sha256, sha384, sha512)
            storage_backend: Optional persistent storage
        """
        self.hash_algorithm = hash_algorithm
        self.storage = storage_backend

        # In-memory storage for development
        self._records: Dict[str, ProvenanceRecord] = {}
        self._chains: Dict[str, ProvenanceChain] = {}
        self._artifacts: Dict[str, Artifact] = {}

        # Current chain tracking
        self._current_chain_id: Optional[str] = None

    def hash_data(self, data: Any) -> str:
        """
        Create cryptographic hash of data.

        Args:
            data: Any data to hash (will be JSON serialized)

        Returns:
            Hex string of hash
        """
        if data is None:
            serialized = b""
        elif isinstance(data, bytes):
            serialized = data
        elif isinstance(data, str):
            serialized = data.encode("utf-8")
        else:
            try:
                serialized = json.dumps(
                    data, sort_keys=True, default=str
                ).encode("utf-8")
            except (TypeError, ValueError):
                serialized = str(data).encode("utf-8")

        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(serialized)
        return hasher.hexdigest()

    def record(
        self,
        actor_id: str,
        actor_type: Union[ActorType, str],
        action: str,
        inputs_hash: str,
        outputs_hash: str,
        tool_id: Optional[str] = None,
        tool_version: Optional[str] = None,
        parent_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Create a provenance record for an action.

        Args:
            actor_id: ID of entity performing action
            actor_type: Type of actor
            action: Description of action
            inputs_hash: Hash of inputs
            outputs_hash: Hash of outputs
            tool_id: Tool used (if any)
            tool_version: Version of tool
            parent_id: Parent provenance record
            execution_id: Overall execution context
            metadata: Additional metadata

        Returns:
            Created ProvenanceRecord
        """
        if isinstance(actor_type, str):
            actor_type = ActorType(actor_type)

        record = ProvenanceRecord(
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            tool_id=tool_id,
            tool_version=tool_version,
            parent_provenance_id=parent_id,
            execution_id=execution_id,
            metadata=metadata or {},
        )

        # Store record
        self._records[record.provenance_id] = record

        # Add to current chain if active
        if self._current_chain_id and self._current_chain_id in self._chains:
            chain = self._chains[self._current_chain_id]
            chain.records.append(record)

        logger.debug(f"Provenance recorded: {record.provenance_id} - {action}")

        return record

    def start_chain(self, execution_id: str) -> ProvenanceChain:
        """
        Start a new provenance chain for an execution.

        Args:
            execution_id: ID of the execution

        Returns:
            New ProvenanceChain
        """
        chain = ProvenanceChain(execution_id=execution_id)
        self._chains[chain.chain_id] = chain
        self._current_chain_id = chain.chain_id

        logger.info(f"Started provenance chain: {chain.chain_id}")
        return chain

    def finalize_chain(self, chain_id: Optional[str] = None) -> ProvenanceChain:
        """
        Finalize a provenance chain and compute root hash.

        Args:
            chain_id: Chain to finalize (or current chain)

        Returns:
            Finalized ProvenanceChain
        """
        chain_id = chain_id or self._current_chain_id
        if not chain_id or chain_id not in self._chains:
            raise ValueError(f"Chain not found: {chain_id}")

        chain = self._chains[chain_id]

        # Compute root hash from all record hashes
        all_hashes = [r.inputs_hash + r.outputs_hash for r in chain.records]
        chain.root_hash = self.hash_data(all_hashes)
        chain.finalized_at = datetime.utcnow()

        # Clear current chain
        if self._current_chain_id == chain_id:
            self._current_chain_id = None

        logger.info(f"Finalized provenance chain: {chain_id}, root_hash={chain.root_hash[:16]}...")

        return chain

    def get_record(self, provenance_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID."""
        return self._records.get(provenance_id)

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """Get a provenance chain by ID."""
        return self._chains.get(chain_id)

    def get_chain_by_execution(self, execution_id: str) -> Optional[ProvenanceChain]:
        """Get provenance chain for an execution."""
        for chain in self._chains.values():
            if chain.execution_id == execution_id:
                return chain
        return None

    def get_records_by_actor(self, actor_id: str) -> List[ProvenanceRecord]:
        """Get all records created by an actor."""
        return [r for r in self._records.values() if r.actor_id == actor_id]

    def get_records_by_tool(self, tool_id: str) -> List[ProvenanceRecord]:
        """Get all records for a specific tool."""
        return [r for r in self._records.values() if r.tool_id == tool_id]

    def verify_record(self, record: ProvenanceRecord) -> bool:
        """
        Verify a provenance record exists and is unchanged.

        Args:
            record: Record to verify

        Returns:
            True if record is valid
        """
        stored = self._records.get(record.provenance_id)
        if not stored:
            return False

        # Compare hashes
        return (
            stored.inputs_hash == record.inputs_hash
            and stored.outputs_hash == record.outputs_hash
        )

    def verify_chain(self, chain_id: str) -> bool:
        """
        Verify integrity of a provenance chain.

        Args:
            chain_id: Chain to verify

        Returns:
            True if chain is valid and unmodified
        """
        chain = self._chains.get(chain_id)
        if not chain:
            return False

        if not chain.finalized_at:
            return True  # Not finalized, cannot verify

        # Recompute root hash
        all_hashes = [r.inputs_hash + r.outputs_hash for r in chain.records]
        computed_hash = self.hash_data(all_hashes)

        return computed_hash == chain.root_hash

    def create_artifact(
        self,
        artifact_type: ArtifactType,
        content: Any,
        created_by: str,
        provenance_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Create a tracked artifact.

        Args:
            artifact_type: Type of artifact
            content: Artifact content
            created_by: Actor creating artifact
            provenance_id: Associated provenance record
            metadata: Additional metadata

        Returns:
            Created Artifact
        """
        content_hash = self.hash_data(content)

        artifact = Artifact(
            artifact_type=artifact_type,
            content_hash=content_hash,
            content=content,
            created_by=created_by,
            provenance_id=provenance_id,
            metadata=metadata or {},
        )

        self._artifacts[artifact.artifact_id] = artifact

        logger.debug(f"Artifact created: {artifact.artifact_id} ({artifact_type.value})")

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)

    def verify_artifact(self, artifact: Artifact) -> bool:
        """Verify artifact integrity."""
        if artifact.content is None:
            return False

        computed_hash = self.hash_data(artifact.content)
        return computed_hash == artifact.content_hash

    def export_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Export a provenance chain for external storage or audit.

        Args:
            chain_id: Chain to export

        Returns:
            Serializable dict of chain data
        """
        chain = self._chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain not found: {chain_id}")

        return {
            "chain_id": chain.chain_id,
            "execution_id": chain.execution_id,
            "records": [r.model_dump() for r in chain.records],
            "root_hash": chain.root_hash,
            "created_at": chain.created_at.isoformat(),
            "finalized_at": chain.finalized_at.isoformat() if chain.finalized_at else None,
            "is_valid": chain.is_valid,
            "record_count": len(chain.records),
        }

    def import_chain(self, data: Dict[str, Any]) -> ProvenanceChain:
        """
        Import a provenance chain from external data.

        Args:
            data: Chain data to import

        Returns:
            Imported ProvenanceChain
        """
        records = [ProvenanceRecord(**r) for r in data.get("records", [])]

        chain = ProvenanceChain(
            chain_id=data["chain_id"],
            execution_id=data["execution_id"],
            records=records,
            root_hash=data.get("root_hash", ""),
            is_valid=data.get("is_valid", True),
        )

        if data.get("created_at"):
            chain.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("finalized_at"):
            chain.finalized_at = datetime.fromisoformat(data["finalized_at"])

        self._chains[chain.chain_id] = chain

        # Also store individual records
        for record in records:
            self._records[record.provenance_id] = record

        return chain

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance tracking statistics."""
        return {
            "total_records": len(self._records),
            "total_chains": len(self._chains),
            "total_artifacts": len(self._artifacts),
            "active_chain": self._current_chain_id,
            "records_by_actor_type": self._count_by_actor_type(),
            "records_by_tool": self._count_by_tool(),
        }

    def _count_by_actor_type(self) -> Dict[str, int]:
        """Count records by actor type."""
        counts: Dict[str, int] = {}
        for record in self._records.values():
            key = record.actor_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_tool(self) -> Dict[str, int]:
        """Count records by tool."""
        counts: Dict[str, int] = {}
        for record in self._records.values():
            if record.tool_id:
                counts[record.tool_id] = counts.get(record.tool_id, 0) + 1
        return counts
