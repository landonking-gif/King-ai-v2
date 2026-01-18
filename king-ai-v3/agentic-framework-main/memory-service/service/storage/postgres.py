"""
PostgreSQL Storage Adapter for Structured Data.

Module: memory-service/service/storage/postgres.py
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
import json

from sqlmodel import Session, create_engine, select, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
import anyio

from ..config import Settings
from ..models import ProvenanceLog, ArtifactRecord, Artifact


class PostgresAdapter:
    """PostgreSQL adapter for structured artifact and provenance storage."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize Postgres adapter.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        # Convert sync URL to async URL for asyncpg
        async_url = settings.postgres_url.replace("postgresql://", "postgresql+asyncpg://")
        self.engine: AsyncEngine = create_async_engine(async_url, echo=False, future=True)

    async def init_db(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hex digest of hash
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def create_provenance_log(
        self,
        artifact_id: str,
        actor_id: str,
        actor_type: str,
        inputs_hash: str,
        outputs_hash: str,
        tool_ids: List[str],
        parent_artifact_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create provenance log entry (append-only).

        Args:
            artifact_id: ID of artifact being logged
            actor_id: ID of actor creating artifact
            actor_type: Type of actor (subagent, lead_agent, skill)
            inputs_hash: Hash of input data
            outputs_hash: Hash of output data
            tool_ids: List of tool IDs used
            parent_artifact_ids: List of parent artifact IDs
            metadata: Additional metadata

        Returns:
            Provenance log ID
        """
        async with AsyncSession(self.engine) as session:
            log = ProvenanceLog(
                artifact_id=artifact_id,
                actor_id=actor_id,
                actor_type=actor_type,
                inputs_hash=inputs_hash,
                outputs_hash=outputs_hash,
                tool_ids=tool_ids,
                parent_artifact_ids=parent_artifact_ids,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log.id

    async def get_provenance_chain(
        self, artifact_id: str, max_depth: Optional[int] = None
    ) -> List[ProvenanceLog]:
        """
        Get complete provenance chain for an artifact.

        Args:
            artifact_id: Starting artifact ID
            max_depth: Maximum depth to traverse (uses config default if None)

        Returns:
            List of provenance log entries in chronological order
        """
        max_depth = max_depth or self.settings.max_provenance_depth
        chain: List[ProvenanceLog] = []
        visited = set()
        current_ids = [artifact_id]

        async with AsyncSession(self.engine) as session:
            for _ in range(max_depth):
                if not current_ids:
                    break

                # Get logs for current artifact IDs
                statement = select(ProvenanceLog).where(ProvenanceLog.artifact_id.in_(current_ids))
                result = await session.exec(statement)
                logs = result.all()

                if not logs:
                    break

                # Add to chain and collect parent IDs
                next_ids = []
                for log in logs:
                    if log.artifact_id not in visited:
                        chain.append(log)
                        visited.add(log.artifact_id)
                        next_ids.extend(log.parent_artifact_ids)

                current_ids = [aid for aid in next_ids if aid not in visited]

        # Sort by created_at
        chain.sort(key=lambda x: x.created_at)
        return chain

    async def create_artifact_record(
        self,
        artifact: Artifact,
        content_hash: str,
        embedding_ref: Optional[str] = None,
        cold_storage_ref: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> str:
        """
        Create artifact record in database.

        Args:
            artifact: Artifact to store
            content_hash: Hash of artifact content
            embedding_ref: Reference to vector embedding
            cold_storage_ref: Reference to cold storage location
            token_count: Estimated token count

        Returns:
            Artifact ID
        """
        async with AsyncSession(self.engine) as session:
            record = ArtifactRecord(
                id=artifact.id or self._generate_id(),
                artifact_type=artifact.artifact_type.value,
                content_hash=content_hash,
                safety_class=artifact.safety_class.value,
                created_by=artifact.created_by,
                created_at=artifact.created_at or datetime.utcnow(),
                metadata=artifact.metadata,
                embedding_ref=embedding_ref,
                cold_storage_ref=cold_storage_ref,
                session_id=artifact.session_id,
                token_count=token_count,
                tags=artifact.tags,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id

    async def get_artifact_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """
        Retrieve artifact record by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Artifact record if found, None otherwise
        """
        async with AsyncSession(self.engine) as session:
            statement = select(ArtifactRecord).where(ArtifactRecord.id == artifact_id)
            result = await session.exec(statement)
            return result.first()

    async def get_artifacts_by_session(self, session_id: str) -> List[ArtifactRecord]:
        """
        Get all artifacts for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of artifact records
        """
        async with AsyncSession(self.engine) as session:
            statement = (
                select(ArtifactRecord)
                .where(ArtifactRecord.session_id == session_id)
                .order_by(ArtifactRecord.created_at.desc())
            )
            result = await session.exec(statement)
            return list(result.all())

    async def get_artifacts_by_type(
        self, artifact_type: str, limit: int = 100
    ) -> List[ArtifactRecord]:
        """
        Get artifacts by type.

        Args:
            artifact_type: Type of artifact
            limit: Maximum number of results

        Returns:
            List of artifact records
        """
        async with AsyncSession(self.engine) as session:
            statement = (
                select(ArtifactRecord)
                .where(ArtifactRecord.artifact_type == artifact_type)
                .order_by(ArtifactRecord.created_at.desc())
                .limit(limit)
            )
            result = await session.exec(statement)
            return list(result.all())

    async def delete_artifact_record(self, artifact_id: str) -> bool:
        """
        Delete artifact record (soft delete - provenance preserved).

        Args:
            artifact_id: Artifact identifier

        Returns:
            True if deleted, False if not found
        """
        async with AsyncSession(self.engine) as session:
            statement = select(ArtifactRecord).where(ArtifactRecord.id == artifact_id)
            result = await session.exec(statement)
            record = result.first()

            if record:
                await session.delete(record)
                await session.commit()
                return True
            return False

    async def get_session_token_count(self, session_id: str) -> int:
        """
        Calculate total token count for a session.

        Args:
            session_id: Session identifier

        Returns:
            Total token count
        """
        artifacts = await self.get_artifacts_by_session(session_id)
        return sum(a.token_count or 0 for a in artifacts)

    def _generate_id(self) -> str:
        """Generate unique artifact ID."""
        import uuid

        return str(uuid.uuid4())
