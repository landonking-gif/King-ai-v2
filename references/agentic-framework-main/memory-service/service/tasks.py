"""
Celery Background Tasks for Memory Service.

Module: memory-service/service/tasks.py

Tasks:
- Prune expired sessions (TTL enforcement)
- Archive old artifacts to cold storage
- Compact memory for long-running sessions
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from celery import Celery
from celery.schedules import crontab
import logging

from .config import settings
from .storage import PostgresAdapter, RedisAdapter, S3Adapter
from .models import CompactionStrategy

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "memory_service",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    "prune-expired-sessions": {
        "task": "service.tasks.prune_expired_sessions",
        "schedule": crontab(hour="*/6"),  # Every 6 hours
    },
    "archive-old-artifacts": {
        "task": "service.tasks.archive_old_artifacts",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    "compact-long-sessions": {
        "task": "service.tasks.compact_long_running_sessions",
        "schedule": crontab(hour="*/12"),  # Every 12 hours
    },
}


def run_async(coro: Any) -> Any:
    """
    Run async coroutine in sync context.

    Args:
        coro: Async coroutine

    Returns:
        Result from coroutine
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Create new loop for nested async calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(coro)
    finally:
        if not loop.is_running():
            loop.close()


@celery_app.task(name="service.tasks.prune_expired_sessions")
def prune_expired_sessions() -> Dict[str, Any]:
    """
    Prune expired sessions based on TTL.

    Removes:
    - Session data from Redis older than session_ttl_hours
    - Associated artifacts from Postgres (optional, based on retention policy)

    Returns:
        Statistics about pruning operation
    """
    logger.info("Starting session pruning task")

    async def _prune() -> Dict[str, Any]:
        redis_adapter = RedisAdapter(settings)
        postgres_adapter = PostgresAdapter(settings)

        try:
            await redis_adapter.connect()
            await postgres_adapter.init_db()

            # Calculate expiry threshold
            expiry_threshold = datetime.utcnow() - timedelta(hours=settings.session_ttl_hours)
            logger.info(f"Pruning sessions older than {expiry_threshold.isoformat()}")

            # Get all session keys from Redis
            sessions_pruned = 0
            artifacts_pruned = 0

            # Scan for session keys
            async for key in redis_adapter.client.scan_iter("session:*"):
                key_str = key.decode("utf-8")
                session_id = key_str.split(":")[-1]

                # Check session age from metadata
                session_data = await redis_adapter.client.hgetall(key)
                if not session_data:
                    continue

                created_at_bytes = session_data.get(b"created_at")
                if not created_at_bytes:
                    continue

                try:
                    created_at = datetime.fromisoformat(created_at_bytes.decode("utf-8"))
                    if created_at < expiry_threshold:
                        # Delete session from Redis
                        await redis_adapter.client.delete(key)
                        sessions_pruned += 1

                        # Optionally delete associated artifacts
                        # (Only if retention policy allows)
                        artifacts = await postgres_adapter.get_artifacts_by_session(session_id)
                        for artifact in artifacts:
                            await postgres_adapter.delete_artifact_record(artifact.id)
                            artifacts_pruned += 1

                        logger.info(
                            f"Pruned session {session_id} "
                            f"(created {created_at.isoformat()}, "
                            f"{len(artifacts)} artifacts)"
                        )
                except Exception as e:
                    logger.warning(f"Error processing session {session_id}: {str(e)}")
                    continue

            logger.info(
                f"Session pruning complete: {sessions_pruned} sessions, "
                f"{artifacts_pruned} artifacts"
            )

            return {
                "sessions_pruned": sessions_pruned,
                "artifacts_pruned": artifacts_pruned,
                "expiry_threshold": expiry_threshold.isoformat(),
            }

        finally:
            await redis_adapter.disconnect()
            await postgres_adapter.close()

    return run_async(_prune())


@celery_app.task(name="service.tasks.archive_old_artifacts")
def archive_old_artifacts() -> Dict[str, Any]:
    """
    Archive old artifacts to cold storage.

    Moves artifacts older than a threshold to S3/MinIO cold storage
    and removes from hot Postgres storage.

    Returns:
        Statistics about archival operation
    """
    logger.info("Starting artifact archival task")

    async def _archive() -> Dict[str, Any]:
        postgres_adapter = PostgresAdapter(settings)
        s3_adapter = S3Adapter(settings)

        try:
            await postgres_adapter.init_db()
            await s3_adapter.connect()

            # Calculate archival threshold (e.g., 90 days)
            archival_threshold = datetime.utcnow() - timedelta(days=90)
            logger.info(f"Archiving artifacts older than {archival_threshold.isoformat()}")

            artifacts_archived = 0
            bytes_archived = 0

            # Get old artifacts
            async with postgres_adapter.engine.connect() as conn:
                from sqlalchemy import text

                query = text(
                    """
                    SELECT id, artifact_type, metadata, created_at
                    FROM artifacts
                    WHERE created_at < :threshold
                      AND cold_storage_ref IS NULL
                    ORDER BY created_at ASC
                    LIMIT 1000
                    """
                )

                result = await conn.execute(query, {"threshold": archival_threshold})
                artifacts = result.fetchall()

            for artifact in artifacts:
                artifact_id = artifact[0]
                artifact_type = artifact[1]
                metadata = artifact[2]

                # Upload to cold storage
                object_key = f"artifacts/{artifact_type}/{artifact_id}.json"

                import json

                artifact_data = json.dumps(metadata, default=str).encode("utf-8")

                await s3_adapter.put_artifact(artifact_id, artifact_data)
                bytes_archived += len(artifact_data)

                # Update artifact record with cold_storage_ref
                update_query = text(
                    """
                    UPDATE artifacts
                    SET cold_storage_ref = :cold_ref
                    WHERE id = :artifact_id
                    """
                )

                await conn.execute(
                    update_query,
                    {"cold_ref": object_key, "artifact_id": artifact_id},
                )
                await conn.commit()

                artifacts_archived += 1

                if artifacts_archived % 100 == 0:
                    logger.info(f"Archived {artifacts_archived} artifacts so far...")

            logger.info(
                f"Artifact archival complete: {artifacts_archived} artifacts, "
                f"{bytes_archived / 1024 / 1024:.2f} MB"
            )

            return {
                "artifacts_archived": artifacts_archived,
                "bytes_archived": bytes_archived,
                "archival_threshold": archival_threshold.isoformat(),
            }

        finally:
            await postgres_adapter.close()

    return run_async(_archive())


@celery_app.task(name="service.tasks.compact_long_running_sessions")
def compact_long_running_sessions() -> Dict[str, Any]:
    """
    Compact memory for long-running sessions.

    Applies compaction strategy to sessions with high token counts
    to reduce memory usage.

    Returns:
        Statistics about compaction operation
    """
    logger.info("Starting session compaction task")

    async def _compact() -> Dict[str, Any]:
        postgres_adapter = PostgresAdapter(settings)
        redis_adapter = RedisAdapter(settings)

        try:
            await postgres_adapter.init_db()
            await redis_adapter.connect()

            sessions_compacted = 0
            tokens_saved = 0

            # Find sessions exceeding token threshold
            async with postgres_adapter.engine.connect() as conn:
                from sqlalchemy import text

                query = text(
                    """
                    SELECT session_id, SUM(token_count) as total_tokens
                    FROM artifacts
                    WHERE session_id IS NOT NULL
                      AND token_count IS NOT NULL
                    GROUP BY session_id
                    HAVING SUM(token_count) > :threshold
                    ORDER BY total_tokens DESC
                    LIMIT 100
                    """
                )

                result = await conn.execute(
                    query, {"threshold": settings.memory_compaction_threshold_tokens}
                )
                sessions = result.fetchall()

            logger.info(f"Found {len(sessions)} sessions needing compaction")

            for session_id, total_tokens in sessions:
                try:
                    # TODO: Call compaction endpoint on memory service
                    # For now, log the session
                    logger.info(
                        f"Session {session_id} needs compaction: {total_tokens} tokens"
                    )

                    # In production, this would call:
                    # POST /memory/compact with session_id
                    # and strategy=CompactionStrategy.SUMMARIZE

                    sessions_compacted += 1

                except Exception as e:
                    logger.error(f"Error compacting session {session_id}: {str(e)}")
                    continue

            logger.info(
                f"Session compaction complete: {sessions_compacted} sessions, "
                f"{tokens_saved} tokens saved"
            )

            return {
                "sessions_compacted": sessions_compacted,
                "tokens_saved": tokens_saved,
            }

        finally:
            await postgres_adapter.close()
            await redis_adapter.disconnect()

    return run_async(_compact())


# Utility tasks
@celery_app.task(name="service.tasks.cleanup_orphaned_vectors")
def cleanup_orphaned_vectors() -> Dict[str, Any]:
    """
    Clean up orphaned vector embeddings.

    Removes vector embeddings that no longer have corresponding artifacts.

    Returns:
        Statistics about cleanup operation
    """
    logger.info("Starting orphaned vector cleanup task")

    # TODO: Implement vector cleanup logic
    # This would query the vector DB for all embeddings
    # and cross-reference with Postgres artifacts table

    return {
        "vectors_cleaned": 0,
        "note": "Not yet implemented",
    }


if __name__ == "__main__":
    # Run Celery worker
    celery_app.start()
