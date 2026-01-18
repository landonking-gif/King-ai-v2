# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-01-01

### Added

#### Core Framework
- Multi-agent orchestration platform with typed artifacts
- Workflow engine with YAML-based manifest support
- Subagent lifecycle management with context isolation
- Memory service with multi-tier storage (Redis, Postgres, Milvus, S3)
- MCP gateway for standardized tool access
- Code executor for sandboxed skill execution

#### LLM Adapters
- Anthropic Claude adapter (claude-sonnet-4, claude-opus-4)
- OpenAI GPT adapter (gpt-4o, gpt-4o-mini)
- Azure OpenAI adapter
- Google Gemini adapter (gemini-2.0-flash)
- Ollama adapter for local inference
- vLLM adapter for optimized local inference

#### Skills System
- Skill registry with auto-discovery
- Dual-format support (native and Anthropic marketplace)
- JIT (Just-In-Time) handler loading
- Safety flags and approval workflows
- Example skills: text_summarize, deep-research

#### MCP Integration
- MCP tool binding system
- Wildcard tool patterns (server:*)
- Scoped access control
- Rate limiting and audit logging
- Tool invocation via HTTP gateway

#### CLI Tool (kautilya)
- Agent creation and management
- Skill scaffolding and import
- LLM configuration
- Manifest generation and validation
- Workflow execution

#### Artifacts
- Typed artifact system with JSON Schema validation
- Provenance tracking (actor, tools, timestamps)
- Built-in artifact types:
  - research_snippet
  - claim_verification
  - code_patch
  - synthesis_result

#### Workflows
- Sequential step execution
- Input/output resolution between steps
- Retry logic with exponential backoff
- Timeout enforcement
- Artifact validation at each step

#### Human-in-the-Loop
- Approval manager for sensitive operations
- Priority-based approval queues
- Automatic expiry handling
- Blocking wait for approval

### Infrastructure
- Docker Compose for local development
- FastAPI-based service architecture
- OpenTelemetry instrumentation
- Prometheus metrics
- Structured logging

### Documentation
- Architecture overview
- Getting started guide
- API reference documentation
- Example projects
- Skill development guide

### Developer Experience
- Black code formatting (100 char line length)
- mypy strict type checking
- pytest test suite
- Comprehensive .gitignore
- Development environment setup scripts

## [0.1.0] - 2023-12-01

### Added
- Initial project structure
- Basic orchestrator prototype
- Simple subagent spawning
- Mock LLM adapter for testing

---

[Unreleased]: https://github.com/your-org/agentic-framework/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/agentic-framework/releases/tag/v1.0.0
[0.1.0]: https://github.com/your-org/agentic-framework/releases/tag/v0.1.0
