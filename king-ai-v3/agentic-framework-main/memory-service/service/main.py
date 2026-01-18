"""
Memory Service FastAPI Application.

Module: memory-service/service/main.py

Provides APIs for:
- compact_commit(artifact) -> memory_id
- query_top_k(embedding) -> items
- get_provenance(id) -> provenance_chain
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import time
import hashlib
import json
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import anyio
import httpx

from .config import settings
from .models import (
    Artifact,
    ArtifactType,
    SafetyClass,
    CommitRequest,
    CommitResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ProvenanceChain,
    ProvenanceEntry,
    CompactionRequest,
    CompactionResponse,
    CompactionStrategy,
)
from .storage import (
    RedisAdapter,
    PostgresAdapter,
    get_vector_adapter,
    S3Adapter,
)
from .embedding import EmbeddingGenerator, TokenBudgetManager


# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Initialize FastAPI app
app = FastAPI(
    title="Memory & Context Service",
    description="Memory service for agentic framework with provenance tracking",
    version="1.0.0",
)


# Global adapter instances
redis_adapter: Optional[RedisAdapter] = None
postgres_adapter: Optional[PostgresAdapter] = None
vector_adapter: Optional[Any] = None
s3_adapter: Optional[S3Adapter] = None
embedding_generator: Optional[EmbeddingGenerator] = None
token_budget_manager: Optional[TokenBudgetManager] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize all storage adapters on startup."""
    global redis_adapter, postgres_adapter, vector_adapter, s3_adapter
    global embedding_generator, token_budget_manager

    # Initialize Redis
    redis_adapter = RedisAdapter(settings)
    await redis_adapter.connect()

    # Initialize Postgres
    postgres_adapter = PostgresAdapter(settings)
    await postgres_adapter.init_db()

    # Initialize Vector DB
    vector_adapter = get_vector_adapter(settings)
    await vector_adapter.connect()
    await vector_adapter.create_collection("artifacts", settings.embedding_dimension)

    # Initialize S3/MinIO
    s3_adapter = S3Adapter(settings)
    await s3_adapter.connect()

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(settings)
    await embedding_generator.initialize()

    # Initialize token budget manager
    token_budget_manager = TokenBudgetManager(settings)

    print(f"Memory service started on {settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up connections on shutdown."""
    if redis_adapter:
        await redis_adapter.disconnect()
    if postgres_adapter:
        await postgres_adapter.close()
    if vector_adapter:
        await vector_adapter.disconnect()
    if s3_adapter:
        await s3_adapter.disconnect()

    print("Memory service shut down")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "memory-service"}


@app.post("/memory/commit", response_model=CommitResponse, status_code=status.HTTP_201_CREATED)
async def commit_artifact(request: CommitRequest) -> CommitResponse:
    """
    Commit artifact to memory with provenance logging.

    This is the primary API: compact_commit(artifact) -> memory_id
    """
    if not all([postgres_adapter, embedding_generator]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage services not initialized",
        )

    artifact = request.artifact
    created_at = datetime.utcnow()

    try:
        # Compute content hash
        content_hash = _compute_hash(artifact.content)

        # Compute input/output hashes for provenance
        inputs_hash = _compute_hash(
            {
                "artifact": artifact.dict(),
                "actor_id": request.actor_id,
                "tool_ids": request.tool_ids,
            }
        )
        outputs_hash = content_hash

        # Generate embedding if requested
        embedding_ref = None
        token_count = None
        if request.generate_embedding:
            embedding, token_count = await embedding_generator.generate_artifact_embedding(
                artifact.content
            )

            # Store in vector DB
            metadata = {
                "artifact_id": artifact.id,
                "artifact_type": artifact.artifact_type.value,
                "safety_class": artifact.safety_class.value,
                "created_by": artifact.created_by,
                "session_id": artifact.session_id or "",
            }
            await vector_adapter.insert_vectors(
                collection_name="artifacts",
                ids=[artifact.id],
                vectors=[embedding],
                metadata=[metadata],
            )
            embedding_ref = f"artifacts/{artifact.id}"

        # Store in cold storage if requested
        cold_storage_ref = None
        if request.store_in_cold and s3_adapter:
            cold_storage_ref = await s3_adapter.store_artifact(artifact.id, artifact.content)

        # Create artifact record in Postgres
        artifact_id = await postgres_adapter.create_artifact_record(
            artifact=artifact,
            content_hash=content_hash,
            embedding_ref=embedding_ref,
            cold_storage_ref=cold_storage_ref,
            token_count=token_count,
        )

        # Create provenance log (append-only)
        provenance_id = await postgres_adapter.create_provenance_log(
            artifact_id=artifact_id,
            actor_id=request.actor_id,
            actor_type=request.actor_type,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            tool_ids=request.tool_ids,
            parent_artifact_ids=artifact.parent_artifact_ids,
            metadata={"commit_time": created_at.isoformat()},
        )

        # Cache in Redis if session ID provided
        if artifact.session_id and redis_adapter:
            await redis_adapter.cache_artifact(artifact_id, artifact.content)
            await redis_adapter.add_to_session_artifacts(artifact.session_id, artifact_id)

        return CommitResponse(
            memory_id=artifact_id,
            artifact_id=artifact_id,
            embedding_generated=request.generate_embedding,
            cold_storage_ref=cold_storage_ref,
            provenance_id=provenance_id,
            created_at=created_at,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to commit artifact: {str(e)}",
        )


@app.post("/memory/query", response_model=QueryResponse)
async def query_artifacts(request: QueryRequest) -> QueryResponse:
    """
    Query similar artifacts using vector search.

    This is the primary API: query_top_k(embedding) -> items
    """
    if not all([vector_adapter, postgres_adapter, embedding_generator]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage services not initialized",
        )

    start_time = time.time()

    try:
        # Get query embedding
        if request.query_embedding:
            query_vector = request.query_embedding
        elif request.query_text:
            query_vector = await embedding_generator.generate_embedding(request.query_text)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query_text or query_embedding must be provided",
            )

        # Build filter expression
        filter_expr = None
        if request.filter_artifact_type:
            filter_expr = f'{{"artifact_type": "{request.filter_artifact_type.value}"}}'
        elif request.filter_session_id:
            filter_expr = f'{{"session_id": "{request.filter_session_id}"}}'

        # Search vector DB
        results = await vector_adapter.search_vectors(
            collection_name="artifacts",
            query_vector=query_vector,
            top_k=request.top_k,
            filter_expr=filter_expr,
        )

        # Filter by minimum similarity
        results = [r for r in results if r[1] >= request.min_similarity]

        # Fetch full artifact records
        query_results = []
        for artifact_id, similarity, metadata in results:
            # Try cache first
            content = None
            if redis_adapter:
                content = await redis_adapter.get_cached_artifact(artifact_id)

            # Fall back to database
            if not content:
                record = await postgres_adapter.get_artifact_record(artifact_id)
                if record:
                    # Fetch from cold storage if needed
                    if record.cold_storage_ref and s3_adapter:
                        content = await s3_adapter.retrieve_artifact(record.cold_storage_ref)
                    else:
                        content = metadata  # Use metadata as fallback

            if content:
                query_results.append(
                    QueryResult(
                        artifact_id=artifact_id,
                        artifact_type=metadata.get("artifact_type", "generic"),
                        content=content,
                        similarity=similarity,
                        metadata=metadata,
                        created_by=metadata.get("created_by", "unknown"),
                        created_at=datetime.utcnow(),  # Should come from record
                    )
                )

        query_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            items=query_results, query_time_ms=query_time_ms, total_candidates=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {str(e)}"
        )


@app.get("/memory/provenance/{artifact_id}", response_model=ProvenanceChain)
async def get_provenance(artifact_id: str) -> ProvenanceChain:
    """
    Get complete provenance chain for an artifact.

    This is the primary API: get_provenance(id) -> provenance_chain
    """
    if not postgres_adapter:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Postgres adapter not initialized",
        )

    try:
        # Get provenance chain
        logs = await postgres_adapter.get_provenance_chain(artifact_id)

        if not logs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No provenance found for artifact {artifact_id}",
            )

        # Convert to response format
        chain = [
            ProvenanceEntry(
                artifact_id=log.artifact_id,
                actor_id=log.actor_id,
                actor_type=log.actor_type,
                inputs_hash=log.inputs_hash,
                outputs_hash=log.outputs_hash,
                tool_ids=log.tool_ids,
                parent_artifact_ids=log.parent_artifact_ids,
                metadata=log.metadata,
                created_at=log.created_at,
            )
            for log in logs
        ]

        # Find root artifacts (those with no parents)
        root_artifacts = [log.artifact_id for log in logs if not log.parent_artifact_ids]

        return ProvenanceChain(
            artifact_id=artifact_id, chain=chain, depth=len(chain), root_artifacts=root_artifacts
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve provenance: {str(e)}",
        )


@app.post("/memory/compact", response_model=CompactionResponse)
async def compact_memory(request: CompactionRequest) -> CompactionResponse:
    """
    Compact memory for a session to reduce token budget.

    Implements memory compaction strategies: summarize, truncate, or none.
    """
    if not all([postgres_adapter, token_budget_manager]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage services not initialized",
        )

    try:
        # Get session artifacts
        artifacts = await postgres_adapter.get_artifacts_by_session(request.session_id)

        if not artifacts:
            return CompactionResponse(
                session_id=request.session_id,
                tokens_before=0,
                tokens_after=0,
                artifacts_compacted=0,
                artifacts_removed=0,
                strategy_used=request.strategy,
            )

        # Calculate current token count
        tokens_before = sum(a.token_count or 0 for a in artifacts)

        # Determine if compaction is needed
        if not token_budget_manager.needs_compaction(tokens_before):
            return CompactionResponse(
                session_id=request.session_id,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                artifacts_compacted=0,
                artifacts_removed=0,
                strategy_used=CompactionStrategy.NONE,
            )

        # Prioritize artifacts
        artifact_list = [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat(),
                "token_count": a.token_count or 0,
                "confidence": a.metadata.get("confidence", 0.5),
            }
            for a in artifacts
        ]
        prioritized = token_budget_manager.prioritize_artifacts(
            artifact_list, request.preserve_artifact_ids
        )

        # Calculate target tokens
        target_tokens = request.target_tokens or token_budget_manager.calculate_target_tokens(
            request.strategy.value
        )

        # Apply compaction strategy
        artifacts_removed = 0
        artifacts_compacted = 0
        current_tokens = tokens_before

        if request.strategy == CompactionStrategy.TRUNCATE:
            # Remove oldest artifacts until under budget
            for artifact in reversed(prioritized):
                if current_tokens <= target_tokens:
                    break
                if artifact["id"] not in request.preserve_artifact_ids:
                    await postgres_adapter.delete_artifact_record(artifact["id"])
                    current_tokens -= artifact["token_count"]
                    artifacts_removed += 1

        elif request.strategy == CompactionStrategy.SUMMARIZE:
            # Implement summarization using Code Executor's text_summarize skill
            logger.info(f"Starting summarization for session {request.session_id}")

            # Get artifacts to compact (exclude preserved ones)
            artifacts_to_compact = [
                a for a in artifacts if a.id not in request.preserve_artifact_ids
            ]

            if not artifacts_to_compact:
                logger.info("No artifacts to compact after filtering preserved IDs")
                tokens_after = current_tokens
            else:
                # Group artifacts by similarity
                groups = await _group_artifacts_by_similarity(artifacts_to_compact)
                logger.info(
                    f"Grouped {len(artifacts_to_compact)} artifacts into {len(groups)} groups"
                )

                summary_artifact_id = None

                for group_idx, group in enumerate(groups):
                    # Extract text content from each artifact in the group
                    combined_texts = []
                    group_artifact_ids = []
                    group_tokens = 0

                    for artifact_record in group:
                        # Extract searchable text from artifact content
                        if artifact_record.metadata:
                            text = embedding_generator.extract_searchable_text(
                                artifact_record.metadata
                            )
                            if text:
                                combined_texts.append(text)
                                group_artifact_ids.append(artifact_record.id)
                                group_tokens += artifact_record.token_count or 0

                    if not combined_texts:
                        logger.warning(f"Group {group_idx} has no extractable text, skipping")
                        continue

                    # Combine all texts with separators
                    combined_text = "\n\n---\n\n".join(combined_texts)

                    # Call code executor to summarize
                    logger.info(
                        f"Summarizing group {group_idx} with {len(combined_texts)} artifacts, "
                        f"{group_tokens} tokens"
                    )
                    summary = await _call_code_executor_summarize(
                        combined_text, summary_length="medium"
                    )

                    if summary:
                        # Create new compacted artifact with summary
                        summary_artifact = Artifact(
                            artifact_type=ArtifactType.GENERIC,
                            content={
                                "summary": summary,
                                "original_artifact_ids": group_artifact_ids,
                                "compaction_metadata": {
                                    "strategy": "summarize",
                                    "original_count": len(group_artifact_ids),
                                    "original_tokens": group_tokens,
                                    "compacted_at": datetime.utcnow().isoformat(),
                                },
                            },
                            safety_class=SafetyClass.INTERNAL,
                            created_by="memory-service-compaction",
                            session_id=request.session_id,
                            parent_artifact_ids=group_artifact_ids,
                            metadata={
                                "compaction_summary": True,
                                "original_count": len(group_artifact_ids),
                            },
                        )

                        # Commit the summary artifact
                        commit_request = CommitRequest(
                            artifact=summary_artifact,
                            actor_id="memory-service",
                            actor_type="system",
                            tool_ids=["text_summarize"],
                            generate_embedding=True,
                            store_in_cold=False,
                        )

                        commit_response = await commit_artifact(commit_request)
                        summary_artifact_id = commit_response.artifact_id

                        # Calculate token savings
                        summary_tokens = embedding_generator.count_tokens(summary)
                        tokens_saved = group_tokens - summary_tokens

                        logger.info(
                            f"Created summary artifact {summary_artifact_id}: "
                            f"{group_tokens} -> {summary_tokens} tokens (saved {tokens_saved})"
                        )

                        # Delete the original artifacts
                        for artifact_id in group_artifact_ids:
                            await postgres_adapter.delete_artifact_record(artifact_id)
                            artifacts_removed += 1

                        # Update token counts
                        current_tokens -= tokens_saved
                        artifacts_compacted += len(group_artifact_ids)

                    else:
                        # Summarization failed, fall back to TRUNCATE for this group
                        logger.warning(
                            f"Summarization failed for group {group_idx}, falling back to truncate"
                        )
                        for artifact_record in group:
                            if current_tokens <= target_tokens:
                                break
                            await postgres_adapter.delete_artifact_record(artifact_record.id)
                            current_tokens -= artifact_record.token_count or 0
                            artifacts_removed += 1

                tokens_after = current_tokens

        else:
            tokens_after = current_tokens

        return CompactionResponse(
            session_id=request.session_id,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            artifacts_compacted=artifacts_compacted,
            artifacts_removed=artifacts_removed,
            strategy_used=request.strategy,
            summary_artifact_id=(
                summary_artifact_id if request.strategy == CompactionStrategy.SUMMARIZE else None
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compaction failed: {str(e)}",
        )


@app.get("/memory/session/{session_id}/stats")
async def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Get statistics for a session."""
    if not all([postgres_adapter, redis_adapter]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage services not initialized",
        )

    try:
        artifacts = await postgres_adapter.get_artifacts_by_session(session_id)
        token_count = sum(a.token_count or 0 for a in artifacts)

        return {
            "session_id": session_id,
            "artifact_count": len(artifacts),
            "total_tokens": token_count,
            "needs_compaction": (
                token_budget_manager.needs_compaction(token_count)
                if token_budget_manager
                else False
            ),
            "artifacts_by_type": _count_by_type(artifacts),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _count_by_type(artifacts: List[Any]) -> Dict[str, int]:
    """Count artifacts by type."""
    counts: Dict[str, int] = {}
    for artifact in artifacts:
        artifact_type = artifact.artifact_type
        counts[artifact_type] = counts.get(artifact_type, 0) + 1
    return counts


async def _call_code_executor_summarize(text: str, summary_length: str = "medium") -> Optional[str]:
    """
    Call code executor's text_summarize skill.

    Args:
        text: Text to summarize
        summary_length: Length of summary (short, medium, long)

    Returns:
        Summary text or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.code_exec_url}/skills/execute",
                json={
                    "skill": "text_summarize",
                    "args": {"text": text, "summary_length": summary_length},
                },
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("result", {}).get("summary")
            else:
                logger.error(
                    f"Code executor returned status {response.status_code}: {response.text}"
                )
                return None

    except httpx.TimeoutException:
        logger.error("Code executor request timed out")
        return None
    except httpx.ConnectError:
        logger.error(f"Failed to connect to code executor at {settings.code_exec_url}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling code executor: {str(e)}")
        return None


async def _group_artifacts_by_similarity(
    artifacts: List[Any], threshold: float = 0.7
) -> List[List[Any]]:
    """
    Group artifacts by semantic similarity using embeddings.

    Args:
        artifacts: List of artifact records
        threshold: Similarity threshold for grouping

    Returns:
        List of artifact groups
    """
    if not vector_adapter or not embedding_generator:
        # Fallback: group all artifacts together
        return [artifacts] if artifacts else []

    # Simple grouping: artifacts with embeddings vs without
    with_embeddings = [a for a in artifacts if a.embedding_ref]
    without_embeddings = [a for a in artifacts if not a.embedding_ref]

    groups = []
    if with_embeddings:
        # For now, group all artifacts with embeddings together
        # A more sophisticated approach would use clustering
        groups.append(with_embeddings)

    if without_embeddings:
        groups.append(without_embeddings)

    return groups


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
