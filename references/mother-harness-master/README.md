# Mother-Harness

> Multi-agent orchestration system with multimodal RAG, Redis Stack, and n8n integration

## Overview

Mother-Harness is a three-tiered agent orchestration system designed for complex research, coding, analysis, and design tasks:

- **Mother Orchestrator**: Plans, routes, and synthesizes
- **Specialist Agents**: Execute domain-specific tasks (12+ agents)
- **Redis Stack**: Unified database (tasks, vectors, sessions, memory)
- **Docling**: Multimodal document ingestion
- **n8n**: Workflow automation + agent execution environment

## Documentation

- [Launch Readiness Checklist](docs/launch-readiness.md)
- [Roadmap](docs/roadmap.md)

## Project Structure

```
mother-harness/
├── .claude/                     # Claude configuration
├── .github/                     # GitHub workflows
├── docker-compose.yml           # All services configuration
├── env.example                  # Environment template
├── package.json                 # Root workspace config
├── pnpm-workspace.yaml          # pnpm workspaces
├── tsconfig.json                # Base TypeScript config
├── vitest.config.ts             # Vitest configuration
│
├── docs/                        # Documentation
├── services/
│   ├── shared/                  # Shared types & utilities
│   │   └── src/
│   │       ├── types/           # TypeScript interfaces
│   │       ├── redis/           # Redis client & indexes
│   │       └── validation/      # Zod schemas
│   │
│   ├── orchestrator/            # Mother Orchestrator
│   │   └── src/
│   │       ├── server.ts        # Fastify server
│   │       ├── orchestrator.ts  # Main orchestration logic
│   │       ├── planner.ts       # Task planning
│   │       └── memory/          # Tiered memory system
│   │
│   ├── docling/                 # Document ingestion
│   ├── agents/                  # Agent workers
│   └── dashboard/               # Next.js UI
│
├── n8n-workflows/               # n8n workflow templates
└── project documents/           # Design PDFs
```

## Quick Start

### Prerequisites

- Node.js 20+
- pnpm 8+
- Docker (for Redis Stack on deployment server)

### Development Setup

1. **Clone and install dependencies**
   ```bash
   git clone <repo-url>
   cd mother-harness
   pnpm install
   ```

2. **Build all packages**
   ```bash
   pnpm build
   ```

3. **Copy environment template**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

### Deployment (Remote Server)

1. **Copy files to server**
   ```bash
   scp -r . user@core1:/opt/mother-harness/
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **Verify health**
   ```bash
   curl http://localhost:8000/health
   ```

## Agent Roster

| Agent | Purpose | Local Model | Cloud Model |
|-------|---------|-------------|-------------|
| Orchestrator | Planning, routing, synthesis | gpt-oss:20b | gpt-oss:120b-cloud |
| Researcher | Web research, documentation | gpt-oss:20b | qwen3-next:80b-cloud |
| Coder | Code generation, git ops | gpt-oss:20b | devstral-2:123b-cloud |
| Design | Architecture, UI/UX | gpt-oss:20b | gemini-3-flash-preview-cloud |
| Analyst | Data analysis, visualization | gpt-oss:20b | qwen3-coder:30b-cloud |
| Critic | Verification, security review | gpt-oss:20b | deepseek-v3.1:671b-cloud |
| Skeptic | Business validation | gpt-oss:20b | deepseek-v3.2-cloud |
| RAG | Document retrieval | gpt-oss:20b | (local only) |
| Librarian | Document ingestion | gpt-oss:20b | (local only) |
| Vision | Multimodal, OCR | gpt-oss:20b | gemini-3-flash-preview-cloud |
| Update | Software inventory | gpt-oss:20b | gpt-oss:120b-cloud |
| Toolsmith | Tool wrappers | gpt-oss:20b | (local only) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/ask` | POST | Create and execute task |
| `/api/task/:id` | GET | Get task status |
| `/api/projects` | GET | List user projects |
| `/api/approvals/pending` | GET | Get pending approvals |
| `/api/approvals/:id/respond` | POST | Approve/reject |

## 3-Tier Memory System

1. **Tier 1 - Recent Context**: Last 10 messages (verbatim)
2. **Tier 2 - Session Summaries**: Structured summaries after task completion
3. **Tier 3 - Long-term Memory**: Vector embeddings for semantic retrieval

## License

Private - All rights reserved

---

*Architecture Version: 1.1 - Production-Ready*
