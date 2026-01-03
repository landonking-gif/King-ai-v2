# Architecture

## Summary
Mother-Harness is organized as a multi-service workspace with a central orchestrator, shared libraries, agent workers, ingestion services, and automation workflows. Redis Stack is the primary data store for tasks, vectors, sessions, and memory.

## Service Boundaries
- **services/orchestrator**: Fastify API service that plans, routes, and synthesizes tasks. Owns task lifecycle, planning, and memory workflows.
- **services/shared**: Shared TypeScript types, Redis clients/indexes, and validation schemas used across services.
- **services/agents**: Agent worker implementations for domain-specific execution. Integrates with orchestrator task definitions.
- **services/docling**: Document ingestion pipeline for multimodal sources and OCR.
- **services/dashboard**: Web UI for task monitoring, approvals, and visibility into agent activity.
- **n8n-workflows**: Workflow templates and automation connectors for agent execution and external integrations.

## Data Flow (High-Level)
1. Clients submit tasks to the orchestrator API.
2. Orchestrator stores task state and memory in Redis Stack and routes work to agents.
3. Agents consume shared types/utilities and push results back to Redis.
4. Document ingestion enriches Redis with embeddings and metadata for retrieval.
5. The dashboard and workflows read from the orchestrator APIs to display status and trigger automations.
