"""
Artifact Store.

Persistent storage for typed artifacts with query capabilities.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.artifacts.models import (
    Artifact, ArtifactType, SafetyClass, ProvenanceRecord
)
from src.utils.structured_logging import get_logger

logger = get_logger("artifact_store")


class ArtifactStore:
    """
    Stores and retrieves artifacts with provenance tracking.
    
    Supports:
    - Create/Read/Update artifacts
    - Query by type, business, session, tags
    - Provenance chain traversal
    - Export for compliance
    """
    
    def __init__(self, db_pool=None, storage_path: Optional[Path] = None):
        """
        Initialize artifact store.
        
        Args:
            db_pool: Database connection pool (optional)
            storage_path: File-based storage path (fallback)
        """
        self.db_pool = db_pool
        self.storage_path = storage_path or Path("data/artifacts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for recent artifacts
        self._cache: Dict[str, Artifact] = {}
        self._cache_max_size = 1000
    
    async def store(self, artifact: Artifact) -> str:
        """
        Store an artifact.
        
        Returns:
            Artifact ID
        """
        # Ensure content hash is computed
        if not artifact.content_hash:
            artifact.content_hash = artifact._compute_hash()
        
        # Try database first
        if self.db_pool:
            await self._store_db(artifact)
        else:
            await self._store_file(artifact)
        
        # Update cache
        self._cache[artifact.id] = artifact
        self._trim_cache()
        
        logger.info(
            "Stored artifact",
            artifact_id=artifact.id,
            artifact_type=artifact.artifact_type.value,
            created_by=artifact.created_by,
        )
        
        return artifact.id
    
    async def get(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID."""
        # Check cache first
        if artifact_id in self._cache:
            return self._cache[artifact_id]
        
        # Try database
        if self.db_pool:
            artifact = await self._get_db(artifact_id)
        else:
            artifact = await self._get_file(artifact_id)
        
        if artifact:
            self._cache[artifact_id] = artifact
        
        return artifact
    
    async def query(
        self,
        artifact_type: Optional[ArtifactType] = None,
        business_id: Optional[str] = None,
        session_id: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Artifact]:
        """
        Query artifacts with filters.
        
        Returns:
            List of matching artifacts
        """
        if self.db_pool:
            return await self._query_db(
                artifact_type, business_id, session_id,
                created_by, tags, since, limit
            )
        else:
            return await self._query_files(
                artifact_type, business_id, session_id,
                created_by, tags, since, limit
            )
    
    async def get_lineage(self, artifact_id: str, depth: int = 5) -> List[Artifact]:
        """
        Get the full lineage chain of an artifact.
        
        Args:
            artifact_id: Starting artifact
            depth: Maximum depth to traverse
            
        Returns:
            List of parent artifacts in order
        """
        lineage = []
        current_id = artifact_id
        seen = set()
        
        for _ in range(depth):
            if current_id in seen:
                break
            seen.add(current_id)
            
            artifact = await self.get(current_id)
            if not artifact:
                break
            
            lineage.append(artifact)
            
            if not artifact.parent_artifact_ids:
                break
            
            current_id = artifact.parent_artifact_ids[0]
        
        return lineage
    
    async def get_children(self, artifact_id: str) -> List[Artifact]:
        """Get all artifacts derived from this one."""
        if self.db_pool:
            return await self._get_children_db(artifact_id)
        else:
            return await self._get_children_files(artifact_id)
    
    async def update(self, artifact: Artifact) -> str:
        """
        Update an artifact, creating a new version.
        
        Returns:
            New artifact ID
        """
        # Create new version
        artifact.previous_version_id = artifact.id
        artifact.id = str(__import__('uuid').uuid4())
        artifact.version += 1
        artifact.content_hash = artifact._compute_hash()
        
        return await self.store(artifact)
    
    async def delete(self, artifact_id: str) -> bool:
        """Soft delete an artifact (mark as deleted)."""
        artifact = await self.get(artifact_id)
        if not artifact:
            return False
        
        artifact.metadata["deleted"] = True
        artifact.metadata["deleted_at"] = datetime.utcnow().isoformat()
        await self.store(artifact)
        
        # Remove from cache
        self._cache.pop(artifact_id, None)
        
        return True
    
    async def export_for_business(
        self,
        business_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export all artifacts for a business (compliance/backup)."""
        artifacts = await self.query(business_id=business_id, limit=10000)
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "business_id": business_id,
            "artifact_count": len(artifacts),
            "artifacts": [a.to_dict() for a in artifacts],
        }
        
        return export_data
    
    # Private methods for storage backends
    
    async def _store_db(self, artifact: Artifact) -> None:
        """Store artifact in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO artifacts (
                    id, artifact_type, name, description, content,
                    safety_class, tags, created_by, created_at,
                    parent_artifact_ids, provenance, business_id,
                    session_id, version, previous_version_id,
                    metadata, content_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                         $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (id) DO UPDATE SET
                    content = $5, metadata = $16, content_hash = $17
            """,
                artifact.id,
                artifact.artifact_type.value,
                artifact.name,
                artifact.description,
                json.dumps(artifact.content),
                artifact.safety_class.value,
                artifact.tags,
                artifact.created_by,
                artifact.created_at,
                artifact.parent_artifact_ids,
                json.dumps(artifact.provenance.to_dict()) if artifact.provenance else None,
                artifact.business_id,
                artifact.session_id,
                artifact.version,
                artifact.previous_version_id,
                json.dumps(artifact.metadata),
                artifact.content_hash,
            )
    
    async def _store_file(self, artifact: Artifact) -> None:
        """Store artifact as JSON file."""
        # Organize by business_id if available
        if artifact.business_id:
            path = self.storage_path / artifact.business_id
        else:
            path = self.storage_path / "_system"
        
        path.mkdir(parents=True, exist_ok=True)
        
        file_path = path / f"{artifact.id}.json"
        with open(file_path, 'w') as f:
            json.dump(artifact.to_dict(), f, indent=2, default=str)
    
    async def _get_db(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact from database."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM artifacts WHERE id = $1",
                artifact_id
            )
            if row:
                return self._row_to_artifact(dict(row))
        return None
    
    async def _get_file(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact from file storage."""
        # Search in all subdirectories
        for file_path in self.storage_path.rglob(f"{artifact_id}.json"):
            with open(file_path) as f:
                data = json.load(f)
                return Artifact.from_dict(data)
        return None
    
    async def _query_db(
        self,
        artifact_type: Optional[ArtifactType],
        business_id: Optional[str],
        session_id: Optional[str],
        created_by: Optional[str],
        tags: Optional[List[str]],
        since: Optional[datetime],
        limit: int,
    ) -> List[Artifact]:
        """Query artifacts from database."""
        conditions = []
        params = []
        param_idx = 1
        
        if artifact_type:
            conditions.append(f"artifact_type = ${param_idx}")
            params.append(artifact_type.value)
            param_idx += 1
        
        if business_id:
            conditions.append(f"business_id = ${param_idx}")
            params.append(business_id)
            param_idx += 1
        
        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1
        
        if created_by:
            conditions.append(f"created_by = ${param_idx}")
            params.append(created_by)
            param_idx += 1
        
        if since:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM artifacts
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT {limit}
            """, *params)
            
            return [self._row_to_artifact(dict(row)) for row in rows]
    
    async def _query_files(
        self,
        artifact_type: Optional[ArtifactType],
        business_id: Optional[str],
        session_id: Optional[str],
        created_by: Optional[str],
        tags: Optional[List[str]],
        since: Optional[datetime],
        limit: int,
    ) -> List[Artifact]:
        """Query artifacts from file storage."""
        results = []
        
        # Determine search path
        if business_id:
            search_path = self.storage_path / business_id
            if not search_path.exists():
                return []
        else:
            search_path = self.storage_path
        
        for file_path in search_path.rglob("*.json"):
            if len(results) >= limit:
                break
            
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Apply filters
                if artifact_type and data.get("artifact_type") != artifact_type.value:
                    continue
                if session_id and data.get("session_id") != session_id:
                    continue
                if created_by and data.get("created_by") != created_by:
                    continue
                if since:
                    created = datetime.fromisoformat(data.get("created_at", ""))
                    if created < since:
                        continue
                if tags:
                    if not any(t in data.get("tags", []) for t in tags):
                        continue
                
                results.append(Artifact.from_dict(data))
            except Exception as e:
                logger.warning(f"Error reading artifact file {file_path}: {e}")
        
        # Sort by created_at descending
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results[:limit]
    
    async def _get_children_db(self, artifact_id: str) -> List[Artifact]:
        """Get child artifacts from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM artifacts
                WHERE $1 = ANY(parent_artifact_ids)
                ORDER BY created_at DESC
            """, artifact_id)
            return [self._row_to_artifact(dict(row)) for row in rows]
    
    async def _get_children_files(self, artifact_id: str) -> List[Artifact]:
        """Get child artifacts from file storage."""
        children = []
        for file_path in self.storage_path.rglob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                if artifact_id in data.get("parent_artifact_ids", []):
                    children.append(Artifact.from_dict(data))
            except Exception:
                pass
        return children
    
    def _row_to_artifact(self, row: Dict[str, Any]) -> Artifact:
        """Convert database row to Artifact."""
        if isinstance(row.get("content"), str):
            row["content"] = json.loads(row["content"])
        if isinstance(row.get("metadata"), str):
            row["metadata"] = json.loads(row["metadata"])
        if isinstance(row.get("provenance"), str) and row["provenance"]:
            row["provenance"] = json.loads(row["provenance"])
        return Artifact.from_dict(row)
    
    def _trim_cache(self) -> None:
        """Trim cache to max size."""
        if len(self._cache) > self._cache_max_size:
            # Remove oldest entries (simple LRU approximation)
            excess = len(self._cache) - self._cache_max_size
            keys_to_remove = list(self._cache.keys())[:excess]
            for key in keys_to_remove:
                del self._cache[key]


# Global store instance
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get or create the global artifact store."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store
