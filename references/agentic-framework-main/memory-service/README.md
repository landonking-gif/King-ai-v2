# Memory & Context Service

Enterprise-grade memory service for the Agentic Framework with multi-tier storage, provenance tracking, and intelligent compaction.

## Architecture

The Memory Service implements a five-tier storage architecture:

1. **Redis** - Session storage and hot cache (TTL-based)
2. **PostgreSQL** - Structured artifact metadata and provenance logs
3. **Vector DB** - Semantic search (Milvus for production, ChromaDB for development)
4. **MinIO/S3** - Cold storage for large artifacts
5. **Embedding Generator** - sentence-transformers for vector embeddings

## Core APIs

### 1. Commit Artifact
```python
POST /memory/commit

{
  "artifact": {
    "artifact_type": "research_snippet",
    "content": {...},
    "created_by": "research-agent",
    "session_id": "session-123",
    "tags": ["research"]
  },
  "actor_id": "agent-1",
  "actor_type": "subagent",
  "tool_ids": ["web_search"],
  "generate_embedding": true,
  "store_in_cold": false
}

Response:
{
  "memory_id": "uuid",
  "artifact_id": "uuid",
  "embedding_generated": true,
  "provenance_id": 123,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 2. Query Artifacts
```python
POST /memory/query

{
  "query_text": "agent framework research",
  "top_k": 5,
  "filter_artifact_type": "research_snippet",
  "min_similarity": 0.7
}

Response:
{
  "items": [
    {
      "artifact_id": "uuid",
      "artifact_type": "research_snippet",
      "content": {...},
      "similarity": 0.92,
      "metadata": {...}
    }
  ],
  "query_time_ms": 45.2,
  "total_candidates": 10
}
```

### 3. Get Provenance Chain
```python
GET /memory/provenance/{artifact_id}

Response:
{
  "artifact_id": "uuid",
  "chain": [
    {
      "artifact_id": "uuid",
      "actor_id": "agent-1",
      "actor_type": "subagent",
      "inputs_hash": "sha256...",
      "outputs_hash": "sha256...",
      "tool_ids": ["web_search"],
      "parent_artifact_ids": ["parent-uuid"],
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "depth": 3,
  "root_artifacts": ["root-uuid"]
}
```

### 4. Compact Memory
```python
POST /memory/compact

{
  "session_id": "session-123",
  "strategy": "summarize",
  "target_tokens": 5000,
  "preserve_artifact_ids": ["important-uuid"]
}

Response:
{
  "session_id": "session-123",
  "tokens_before": 12000,
  "tokens_after": 4800,
  "artifacts_compacted": 5,
  "artifacts_removed": 2,
  "strategy_used": "summarize"
}
```

## Setup

### 1. Install Dependencies
```bash
cd agent-framework/memory-service
source ../../.venv/bin/activate  # or create new venv
uv pip install -r ../../requirements.txt
```

### 2. Configure Environment
```bash
cp ../../.env.example .env
# Edit .env with your database credentials
```

Required environment variables:
- `REDIS_URL` - Redis connection string
- `POSTGRES_URL` - PostgreSQL connection string
- `MILVUS_URL` or `CHROMA_PATH` - Vector database
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` - Object storage

### 3. Initialize Database
```bash
# PostgreSQL tables are auto-created on startup
# For production, use Alembic migrations (TODO)
python -m alembic upgrade head
```

### 4. Run Service
```bash
# Development
python service/main.py

# Production
uvicorn service.main:app --host 0.0.0.0 --port 8001 --workers 4
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=service --cov-report=html

# Integration tests (requires running services)
pytest tests/ -m integration
```

## Storage Adapters

### Redis Adapter
- Session data with TTL
- Artifact caching
- Session-to-artifacts mapping

### Postgres Adapter
- Artifact records (structured metadata)
- Provenance logs (append-only)
- Session statistics
- Provenance chain traversal

### Vector DB Adapter
- ChromaDB (development) - file-based, no server required
- Milvus (production) - distributed, scalable
- Cosine similarity search
- Metadata filtering

### S3 Adapter
- MinIO (development/on-prem)
- AWS S3 (cloud production)
- Large artifact storage
- Lifecycle policies

## Provenance Tracking

All artifact commits create append-only provenance logs with:
- **Actor information** - Who created the artifact
- **Input/output hashes** - Deterministic verification
- **Tool IDs** - Which tools were used
- **Parent artifacts** - Lineage tracking
- **Timestamp** - When created

Provenance chains are fully replayable and auditable.

## Memory Compaction

### Strategies

1. **Summarize** - Use LLM to create condensed summaries (TODO: implement)
2. **Truncate** - Remove oldest, least important artifacts
3. **None** - No compaction

### Prioritization
Artifacts are prioritized by:
1. Explicitly preserved IDs
2. Recency (newer = higher priority)
3. Confidence scores
4. Reference count (how many other artifacts reference it)

### Token Budget
- Configurable threshold (default: 8000 tokens)
- Automatic compaction triggers
- Per-session tracking

## Type Checking

```bash
mypy service/ --strict
```

## Formatting

```bash
black service/ tests/ --line-length 100
```

## API Documentation

Start the service and visit:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Production Considerations

1. **Database Migrations** - Use Alembic for schema changes
2. **Connection Pooling** - Configure appropriate pool sizes
3. **Monitoring** - OpenTelemetry integration included
4. **Backup** - Regular backups of Postgres and vector DB
5. **Scaling** - Horizontal scaling via load balancer
6. **Security** - Enable TLS for all connections in production

## Troubleshooting

### Vector DB Connection Issues
- **ChromaDB**: Ensure `CHROMA_PATH` directory is writable
- **Milvus**: Verify service is running on `MILVUS_URL`

### Redis Connection Issues
- Check Redis is running: `redis-cli ping`
- Verify `REDIS_URL` format: `redis://localhost:6379/0`

### Postgres Connection Issues
- Test connection: `psql $POSTGRES_URL`
- Ensure database exists
- Check user permissions

### MinIO Connection Issues
- Verify endpoint reachable: `curl http://localhost:9000/minio/health/live`
- Check credentials match
- Ensure bucket exists (auto-created on startup)

## License

Internal use only - Financial Services Division
