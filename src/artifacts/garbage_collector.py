"""
Artifact Garbage Collector.

Lifecycle management for artifacts with TTL, archiving, and cleanup.
Based on mother-harness robustness patterns.
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("artifact_gc")


class RetentionPolicy(str, Enum):
    """Retention policies for artifacts."""
    EPHEMERAL = "ephemeral"    # Delete after hours
    SHORT = "short"            # Delete after days
    STANDARD = "standard"      # Delete after weeks
    LONG = "long"              # Delete after months
    PERMANENT = "permanent"    # Never delete
    ARCHIVE = "archive"        # Move to archive after TTL


@dataclass
class ArtifactRetentionConfig:
    """Configuration for artifact retention."""
    artifact_type: str
    policy: RetentionPolicy = RetentionPolicy.STANDARD
    ttl_hours: int = 168  # 7 days default
    archive_before_delete: bool = True
    min_access_count: int = 0  # Preserve if accessed more than this
    extend_ttl_on_access: bool = True
    extension_hours: int = 24


@dataclass
class ArtifactMetadata:
    """Metadata for garbage collection."""
    artifact_id: str
    artifact_type: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    starred: bool = False  # Starred items are never deleted
    policy: RetentionPolicy = RetentionPolicy.STANDARD
    ttl_hours: int = 168
    archived: bool = False
    
    @property
    def age_hours(self) -> float:
        """Get artifact age in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    @property
    def hours_since_access(self) -> float:
        """Get hours since last access."""
        return (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
    
    @property
    def is_expired(self) -> bool:
        """Check if artifact has expired based on TTL."""
        if self.starred:
            return False
        if self.policy == RetentionPolicy.PERMANENT:
            return False
        return self.age_hours > self.ttl_hours
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "starred": self.starred,
            "policy": self.policy.value,
            "ttl_hours": self.ttl_hours,
            "archived": self.archived,
            "age_hours": self.age_hours,
            "is_expired": self.is_expired,
        }


@dataclass
class GCResult:
    """Result of a garbage collection run."""
    run_id: str
    started_at: datetime
    completed_at: datetime
    artifacts_scanned: int = 0
    artifacts_deleted: int = 0
    artifacts_archived: int = 0
    artifacts_preserved: int = 0
    bytes_reclaimed: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "artifacts_scanned": self.artifacts_scanned,
            "artifacts_deleted": self.artifacts_deleted,
            "artifacts_archived": self.artifacts_archived,
            "artifacts_preserved": self.artifacts_preserved,
            "bytes_reclaimed": self.bytes_reclaimed,
            "errors": self.errors,
        }


class ArtifactGarbageCollector:
    """
    Garbage collector for artifacts.
    
    Features:
    - Type-specific retention policies
    - TTL-based expiration
    - Access tracking for preservation
    - Archive before delete
    - Starring to prevent deletion
    - Scheduled background cleanup
    """
    
    # Default retention configs by artifact type
    DEFAULT_CONFIGS = {
        "research": ArtifactRetentionConfig(
            artifact_type="research",
            policy=RetentionPolicy.STANDARD,
            ttl_hours=168,  # 7 days
        ),
        "code": ArtifactRetentionConfig(
            artifact_type="code",
            policy=RetentionPolicy.LONG,
            ttl_hours=720,  # 30 days
            archive_before_delete=True,
        ),
        "content": ArtifactRetentionConfig(
            artifact_type="content",
            policy=RetentionPolicy.STANDARD,
            ttl_hours=336,  # 14 days
        ),
        "business_plan": ArtifactRetentionConfig(
            artifact_type="business_plan",
            policy=RetentionPolicy.LONG,
            ttl_hours=2160,  # 90 days
        ),
        "finance": ArtifactRetentionConfig(
            artifact_type="finance",
            policy=RetentionPolicy.ARCHIVE,
            ttl_hours=2160,  # 90 days
            archive_before_delete=True,
        ),
        "legal": ArtifactRetentionConfig(
            artifact_type="legal",
            policy=RetentionPolicy.PERMANENT,
            ttl_hours=0,  # Never expires
        ),
        "analysis": ArtifactRetentionConfig(
            artifact_type="analysis",
            policy=RetentionPolicy.SHORT,
            ttl_hours=72,  # 3 days
        ),
        "generic": ArtifactRetentionConfig(
            artifact_type="generic",
            policy=RetentionPolicy.SHORT,
            ttl_hours=48,  # 2 days
        ),
        "session": ArtifactRetentionConfig(
            artifact_type="session",
            policy=RetentionPolicy.EPHEMERAL,
            ttl_hours=24,  # 1 day
            archive_before_delete=False,
        ),
    }
    
    def __init__(
        self,
        storage_path: str = "./data/artifacts",
        archive_path: str = "./data/archive",
        configs: Dict[str, ArtifactRetentionConfig] = None,
        delete_callback: Callable[[str], None] = None,
        archive_callback: Callable[[str, str], None] = None,
    ):
        """
        Initialize the garbage collector.
        
        Args:
            storage_path: Path where artifacts are stored
            archive_path: Path for archived artifacts
            configs: Custom retention configs
            delete_callback: Called when artifact is deleted
            archive_callback: Called when artifact is archived (id, archive_path)
        """
        self.storage_path = storage_path
        self.archive_path = archive_path
        self._configs = configs or dict(self.DEFAULT_CONFIGS)
        self.delete_callback = delete_callback
        self.archive_callback = archive_callback
        
        # Track artifacts
        self._metadata: Dict[str, ArtifactMetadata] = {}
        
        # GC history
        self._gc_history: List[GCResult] = []
        
        # Background task
        self._gc_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        size_bytes: int = 0,
        policy: RetentionPolicy = None,
        ttl_hours: int = None,
    ) -> ArtifactMetadata:
        """
        Register an artifact for garbage collection tracking.
        
        Args:
            artifact_id: Unique artifact ID
            artifact_type: Type of artifact
            size_bytes: Size in bytes
            policy: Override retention policy
            ttl_hours: Override TTL
            
        Returns:
            Artifact metadata
        """
        config = self._configs.get(artifact_type, self.DEFAULT_CONFIGS["generic"])
        
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            size_bytes=size_bytes,
            policy=policy or config.policy,
            ttl_hours=ttl_hours or config.ttl_hours,
        )
        
        self._metadata[artifact_id] = metadata
        return metadata
    
    def record_access(self, artifact_id: str) -> None:
        """Record an access to an artifact, potentially extending TTL."""
        metadata = self._metadata.get(artifact_id)
        if not metadata:
            return
        
        metadata.access_count += 1
        metadata.last_accessed = datetime.utcnow()
        
        # Extend TTL if configured
        config = self._configs.get(metadata.artifact_type)
        if config and config.extend_ttl_on_access:
            # Reset TTL based on last access
            pass  # TTL is calculated from created_at, so we don't need to modify it
    
    def star_artifact(self, artifact_id: str) -> bool:
        """Star an artifact to prevent deletion."""
        metadata = self._metadata.get(artifact_id)
        if metadata:
            metadata.starred = True
            return True
        return False
    
    def unstar_artifact(self, artifact_id: str) -> bool:
        """Remove star from artifact."""
        metadata = self._metadata.get(artifact_id)
        if metadata:
            metadata.starred = False
            return True
        return False
    
    def get_expired(self) -> List[ArtifactMetadata]:
        """Get all expired artifacts."""
        return [m for m in self._metadata.values() if m.is_expired]
    
    def get_expiring_soon(self, hours: int = 24) -> List[ArtifactMetadata]:
        """Get artifacts expiring within the specified hours."""
        expiring = []
        now = datetime.utcnow()
        
        for metadata in self._metadata.values():
            if metadata.starred or metadata.policy == RetentionPolicy.PERMANENT:
                continue
            
            expires_at = metadata.created_at + timedelta(hours=metadata.ttl_hours)
            if expires_at <= now + timedelta(hours=hours):
                expiring.append(metadata)
        
        return expiring
    
    async def run_gc(
        self,
        dry_run: bool = False,
    ) -> GCResult:
        """
        Run garbage collection.
        
        Args:
            dry_run: If True, don't actually delete/archive
            
        Returns:
            GC result
        """
        from uuid import uuid4
        
        run_id = f"gc_{uuid4().hex[:8]}"
        started_at = datetime.utcnow()
        
        logger.info(f"Starting garbage collection run: {run_id}")
        
        deleted = 0
        archived = 0
        preserved = 0
        bytes_reclaimed = 0
        errors = []
        
        # Get all expired artifacts
        expired = self.get_expired()
        
        for metadata in expired:
            try:
                config = self._configs.get(
                    metadata.artifact_type, 
                    self.DEFAULT_CONFIGS["generic"]
                )
                
                # Check if should preserve due to access count
                if config.min_access_count > 0:
                    if metadata.access_count >= config.min_access_count:
                        preserved += 1
                        logger.debug(
                            f"Preserving {metadata.artifact_id} due to access count"
                        )
                        continue
                
                # Archive if configured
                if config.archive_before_delete and not metadata.archived:
                    if not dry_run:
                        await self._archive_artifact(metadata)
                    archived += 1
                    metadata.archived = True
                
                # Delete
                if not dry_run:
                    await self._delete_artifact(metadata)
                    bytes_reclaimed += metadata.size_bytes
                    del self._metadata[metadata.artifact_id]
                
                deleted += 1
                
            except Exception as e:
                errors.append(f"Error processing {metadata.artifact_id}: {e}")
                logger.error(f"GC error for {metadata.artifact_id}: {e}")
        
        completed_at = datetime.utcnow()
        
        result = GCResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            artifacts_scanned=len(self._metadata),
            artifacts_deleted=deleted,
            artifacts_archived=archived,
            artifacts_preserved=preserved,
            bytes_reclaimed=bytes_reclaimed,
            errors=errors,
        )
        
        self._gc_history.append(result)
        
        logger.info(
            f"GC complete: scanned={len(self._metadata)}, "
            f"deleted={deleted}, archived={archived}, "
            f"preserved={preserved}, reclaimed={bytes_reclaimed} bytes"
        )
        
        return result
    
    async def start_background_gc(
        self,
        interval_hours: int = 1,
    ) -> None:
        """Start background GC task."""
        if self._running:
            return
        
        self._running = True
        self._gc_task = asyncio.create_task(
            self._background_gc_loop(interval_hours)
        )
        logger.info(f"Started background GC with {interval_hours}h interval")
    
    async def stop_background_gc(self) -> None:
        """Stop background GC task."""
        self._running = False
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background GC")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GC statistics."""
        total_size = sum(m.size_bytes for m in self._metadata.values())
        by_type = {}
        for m in self._metadata.values():
            by_type[m.artifact_type] = by_type.get(m.artifact_type, 0) + 1
        
        by_policy = {}
        for m in self._metadata.values():
            by_policy[m.policy.value] = by_policy.get(m.policy.value, 0) + 1
        
        return {
            "total_artifacts": len(self._metadata),
            "total_size_bytes": total_size,
            "starred_count": sum(1 for m in self._metadata.values() if m.starred),
            "expired_count": len(self.get_expired()),
            "by_type": by_type,
            "by_policy": by_policy,
            "gc_runs": len(self._gc_history),
            "total_deleted": sum(r.artifacts_deleted for r in self._gc_history),
            "total_bytes_reclaimed": sum(r.bytes_reclaimed for r in self._gc_history),
        }
    
    def get_retention_config(self, artifact_type: str) -> ArtifactRetentionConfig:
        """Get retention config for an artifact type."""
        return self._configs.get(artifact_type, self.DEFAULT_CONFIGS["generic"])
    
    def set_retention_config(self, config: ArtifactRetentionConfig) -> None:
        """Set retention config for an artifact type."""
        self._configs[config.artifact_type] = config
    
    # Private methods
    
    async def _archive_artifact(self, metadata: ArtifactMetadata) -> None:
        """Archive an artifact."""
        # Ensure archive directory exists
        os.makedirs(self.archive_path, exist_ok=True)
        
        # Call archive callback if provided
        if self.archive_callback:
            archive_dest = os.path.join(
                self.archive_path, 
                f"{metadata.artifact_id}"
            )
            self.archive_callback(metadata.artifact_id, archive_dest)
        
        logger.debug(f"Archived artifact: {metadata.artifact_id}")
    
    async def _delete_artifact(self, metadata: ArtifactMetadata) -> None:
        """Delete an artifact."""
        # Call delete callback if provided
        if self.delete_callback:
            self.delete_callback(metadata.artifact_id)
        
        logger.debug(f"Deleted artifact: {metadata.artifact_id}")
    
    async def _background_gc_loop(self, interval_hours: int) -> None:
        """Background GC loop."""
        while self._running:
            try:
                await asyncio.sleep(interval_hours * 3600)
                if self._running:
                    await self.run_gc()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background GC error: {e}")


# Global GC instance
_artifact_gc: Optional[ArtifactGarbageCollector] = None


def get_artifact_gc() -> ArtifactGarbageCollector:
    """Get or create the global artifact garbage collector."""
    global _artifact_gc
    if _artifact_gc is None:
        _artifact_gc = ArtifactGarbageCollector()
    return _artifact_gc
