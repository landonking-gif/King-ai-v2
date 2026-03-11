# Agentic Framework

> **Enterprise-grade multi-agent orchestration platform for building production-ready LLM workflows**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Release](https://img.shields.io/github/v/release/paragajg/agentic-framework)](https://github.com/paragajg/agentic-framework/releases)

## 📋 Table of Contents

- [What is Agentic Framework?](#-what-is-agentic-framework)
- [Why Agentic Framework?](#-why-agentic-framework)
- [Architecture](#-architecture)
- [Key Concepts](#-key-concepts)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [CLI Usage Guide](#-cli-usage-guide)
- [Multi-Agent Patterns](#-multi-agent-patterns)
- [LLM Provider Support](#-llm-provider-support)
- [Examples](#-examples)
- [Development](#-development)
- [Production Deployment](#-production-deployment)
- [Contributing](#-contributing)
- [FAQ](#-faq)

## 🎯 What is Agentic Framework?

**Agentic Framework** is an open-source, production-ready platform for building complex multi-agent systems powered by Large Language Models. It enables developers to orchestrate multiple specialized AI agents that collaborate to solve complex tasks through a **declarative YAML-based workflow system**.

### Who is this for?

- **Enterprise developers** building scalable AI applications
- **AI engineers** creating multi-step reasoning systems
- **Research teams** experimenting with agent architectures
- **Product teams** integrating LLMs into production workflows

### What problems does it solve?

✅ **Local-first with Ollama** - Runs entirely on your infrastructure with llama3.1:70b, no cloud API costs
✅ **Safe tool execution** - Sandboxed Python skills + MCP gateway for external tools
✅ **Memory management** - Multi-tier storage with automatic compaction
✅ **Auditability** - Full provenance tracking for compliance
✅ **Scalability** - Production-grade architecture with observability
✅ **Optional cloud LLMs** - Switch to Anthropic, OpenAI, Azure, Gemini, or vLLM when needed

## 🤔 Why Agentic Framework?

| Feature | Agentic Framework | LangChain | AutoGen | Other Frameworks |
|---------|-------------------|-----------|---------|------------------|
| **LLM Agnostic** | ✅ 6 providers | ⚠️ Limited | ⚠️ OpenAI-focused | ❌ Vendor-locked |
| **Declarative Workflows** | ✅ YAML manifests | ❌ Code-only | ❌ Code-only | ⚠️ Limited |
| **Typed Artifacts** | ✅ JSON Schema | ❌ No validation | ❌ No validation | ⚠️ Basic |
| **Provenance Tracking** | ✅ Full audit trail | ❌ No | ❌ No | ❌ No |
| **Sandboxed Execution** | ✅ Isolated skills | ⚠️ Unsafe | ⚠️ Unsafe | ❌ No |
| **MCP Integration** | ✅ Native support | ❌ No | ❌ No | ❌ No |
| **Enterprise Features** | ✅ RBAC, approvals | ❌ No | ❌ No | ⚠️ Limited |
| **Memory Management** | ✅ Multi-tier (4 layers) | ⚠️ Basic | ⚠️ Basic | ❌ No |

## 🏗️ Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER / CLIENT                                   │
│                    (CLI, API, Web Interface, SDK)                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LEAD AGENT / ORCHESTRATOR                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  • Workflow Planning & Decomposition                                 │   │
│  │  • YAML Manifest Parser                                              │   │
│  │  • Subagent Task Assignment                                          │   │
│  │  • Artifact Validation & Routing                                     │   │
│  │  • Human-in-the-Loop Approval Gates                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │   SUBAGENT     │  │   SUBAGENT     │  │   SUBAGENT     │
    │   MANAGER      │  │   MANAGER      │  │   MANAGER      │
    └────────────────┘  └────────────────┘  └────────────────┘
            │                   │                   │
            │                   │                   │
    ┌───────┴────────┐  ┌───────┴────────┐  ┌──────┴─────────┐
    │                │  │                │  │                │
    ▼                ▼  ▼                ▼  ▼                ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│Research│  │Verify  │  │  Code  │  │Analysis│  │Synthesis│ │Custom  │
│ Agent  │  │ Agent  │  │ Agent  │  │ Agent  │  │ Agent  │  │ Agent  │
└───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
    │           │           │           │           │           │
    │  ┌────────┴───────────┴───────────┴───────────┴──────┐    │
    │  │         TYPED ARTIFACTS (JSON Schema)              │    │
    │  │  • research_snippet  • claim_verification          │    │
    │  │  • code_patch        • synthesis_result            │    │
    │  └────────────────────────────────────────────────────┘    │
    │                                                             │
    └─────────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  CODE EXECUTOR   │  │   MCP GATEWAY    │  │ MEMORY SERVICE   │
│  (Skills)        │  │   (Tools)        │  │ (4-Tier Storage) │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ • Skill Registry │  │ • Tool Catalog   │  │ • Session (Redis)│
│ • JIT Loading    │  │ • Discovery      │  │ • Vector (Milvus)│
│ • Sandboxed Exec │  │ • Auth & Scopes  │  │ • Struct (Postgres)│
│ • Safety Flags   │  │ • Rate Limiting  │  │ • Cold (S3/MinIO)│
│ • Dual Format:   │  │ • PII Filtering  │  │ • Auto-Compaction│
│   - Native       │  │ • Proxy Runtime  │  │ • Provenance Log │
│   - Anthropic    │  │                  │  │ • Diary/Reflect  │
│ • Ralph Agent 🆕 │  │                  │  │   (Ralph Memory) │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  EXTERNAL WORLD   │
                    ├───────────────────┤
                    │ • File System     │
                    │ • Databases       │
                    │ • Web APIs        │
                    │ • GitHub/Jira     │
                    │ • Slack/Email     │
                    └───────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        CROSS-CUTTING CONCERNS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Observability: OpenTelemetry (Metrics, Traces, Logs)                     │
│  • Security: RBAC, Audit Trails, Policy Enforcement (OPA)                   │
│  • Governance: Approval Workflows, Budget Limits, Guardrails                │
│  • Monitoring: Prometheus, Grafana Dashboards                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: How a Workflow Executes

```
1. USER submits task
   ↓
2. ORCHESTRATOR parses YAML manifest
   ↓
3. ORCHESTRATOR spawns SUBAGENT (e.g., Research Agent)
   ├─ Binds skills: [web_search, summarize]
   ├─ Binds MCP tools: [github.*, firecrawl.*]
   └─ Sets context: isolated LLM session
   ↓
4. SUBAGENT executes task
   ├─ Calls CODE EXECUTOR for deterministic skills
   ├─ Calls MCP GATEWAY for external tools
   ├─ Reads/writes MEMORY SERVICE
   └─ Produces TYPED ARTIFACT (validated by JSON Schema)
   ↓
5. ORCHESTRATOR validates artifact
   ├─ Schema validation (structure)
   ├─ Safety checks (PII, harmful content)
   └─ Provenance logging (who, what, when, why)
   ↓
6. ORCHESTRATOR passes artifact to next SUBAGENT
   ↓
7. Repeat steps 3-6 for each workflow step
   ↓
8. ORCHESTRATOR returns FINAL ARTIFACT to user
   ├─ Stores in MEMORY SERVICE
   └─ Logs provenance chain
```

## 🧩 Key Concepts

### 1. **Typed Artifacts**
Structured, validated data passed between agents:
```python
research_snippet = {
    "id": "uuid",
    "source": {"url": "...", "doc_id": "..."},
    "text": "...",
    "summary": "...",
    "tags": ["ai", "agents"],
    "confidence": 0.95,
    "provenance": {...},  # Full audit trail
    "created_by": "research-agent-01",
    "created_at": "2024-01-01T10:00:00Z"
}
```

### 2. **Skills (Deterministic Functions)**
Sandboxed Python code executed by Code Executor:
- **Native Format**: `skill.yaml` + `schema.json` + `handler.py`
- **Anthropic Format**: `SKILL.md` (marketplace compatible)
- **Hybrid Format**: Both (recommended for sharing)
- **Safety Flags**: `pii_risk`, `file_system`, `network_access`, `side_effect`
- **JIT Loading**: Skills loaded on-demand, cached for performance

### 3. **MCP Tools (External Services)**
Model Context Protocol integration for safe external access:
- **Tool Catalog**: Pre-approved tools (filesystem, github, postgres, slack)
- **Scoped Access**: Fine-grained permissions (e.g., `repo:read`, `issues:write`)
- **Rate Limiting**: Configurable per tool/server
- **PII Filtering**: Automatic scrubbing of sensitive data

### 4. **Capabilities**
What an agent can do:
```yaml
capabilities:
  - web_search      # MCP tool
  - summarize       # Native skill
  - code_execution  # Native skill
  - kb_search       # MCP tool
```

### 5. **Provenance**
Full audit trail for every artifact:
```json
{
  "actor_id": "research-agent-01",
  "actor_type": "subagent",
  "inputs_hash": "sha256:...",
  "outputs_hash": "sha256:...",
  "tool_ids": ["web_search", "summarize"],
  "timestamp": "2024-01-01T10:00:00Z",
  "parent_artifact_id": "uuid"
}
```

### 6. **Memory Tiers**
4-layer storage hierarchy:
- **Session (Redis)**: Hot, fast access for active workflows
- **Vector (Milvus/Chroma)**: Semantic search for retrieval
- **Structured (Postgres)**: Queryable metadata and provenance
- **Cold (S3/MinIO)**: Long-term archival storage

## ✨ Features

### Core Platform
- 🔄 **LLM-Agnostic Architecture**: Swap providers without code changes
- 📝 **YAML-Based Workflows**: Declarative, version-controlled manifests
- 🛠️ **Dual-Format Skills**: Native + Anthropic marketplace compatible
- 🔌 **MCP Integration**: 50+ pre-built tool servers
- 📊 **Full Observability**: OpenTelemetry metrics, traces, logs
- 🔐 **Enterprise Security**: RBAC, audit trails, policy enforcement
- 🎯 **Typed Artifacts**: JSON Schema validated inter-agent communication

### Developer Experience
- 🚀 **kautilya CLI**: Interactive agent/workflow management
- 📚 **Rich Examples**: 4+ complete working examples
- 🧪 **Comprehensive Tests**: 90%+ code coverage
- 📖 **Extensive Docs**: Architecture, API, guides
- 🐳 **Docker Support**: Full-stack compose files
- ☸️ **Kubernetes Ready**: Helm charts for production

### Production Features
- ⚡ **Performance**: JIT skill loading, connection pooling, caching
- 📈 **Scalability**: Horizontal scaling, async execution
- 🔄 **Reliability**: Retries, timeouts, circuit breakers
- 🛡️ **Safety**: Sandboxed execution, PII filtering, guardrails
- 👥 **Collaboration**: Human-in-the-loop approvals
- 📊 **Monitoring**: Prometheus metrics, Grafana dashboards

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Docker & Docker Compose** (for infrastructure services)
- **LLM API Key** (Anthropic, OpenAI, Azure, or Gemini)

### Step 1: Installation

**Option A: Install from Source (Recommended)**
```bash
# Clone repository
git clone https://github.com/paragajg/agentic-framework.git
cd agentic-framework

# Create virtual environment with uv (fast) or venv
# Using uv (recommended - faster):
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Or using standard venv:
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

**Option B: Install from GitHub (pip only)**
```bash
# Install latest release
pip install git+https://github.com/paragajg/agentic-framework.git@v1.0.0

# Or install from main branch
pip install git+https://github.com/paragajg/agentic-framework.git
```

> **Note**: The `kautilya` CLI tool is automatically installed with the framework. No separate installation needed.

### Step 2: Set Up LLM Provider

The system uses **Ollama** for local LLM inference (no API keys required):

```bash
# Install Ollama
# Windows/Mac: Download from https://ollama.ai
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull the required model (in a new terminal)
ollama pull llama3.1:70b

# Verify it's working
ollama run llama3.1:70b "Hello, test message"
```

**Note:** Ralph loop uses GitHub Copilot CLI for code generation:
```bash
# Install GitHub Copilot CLI
npm install -g @githubnext/github-copilot-cli

# Authenticate
gh auth login

# Verify
copilot --version
```

**Optional Cloud Providers** (if you want to use cloud LLMs instead):
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_OPENAI_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Google Gemini
export GEMINI_API_KEY="..."
```

### Step 3: Start Infrastructure (Optional)

```bash
# Start Redis, Postgres, Milvus for production features
docker-compose up -d
```

### Step 4: Verify Installation

```bash
# Check CLI is installed
kautilya --version
# Output: kautilya version 1.0.0

# Test Ollama connection
ollama list
# Output: Should show llama3.1:70b model

# Test LLM connection (optional)
kautilya llm test
# Output: ✓ Connected to Ollama (llama3.1:70b)
```

### Step 5: Run Your First Agent

```bash
# Navigate to example
cd examples/01-simple-agent

# Run the agent
python run.py
```

**Expected Output:**
```
==========================================================
Simple Agent Example
==========================================================

📝 Task: Research the latest trends in AI agents

🤖 Agent Configuration:
   - Role: Research
   - Capabilities: web_search, summarize
   - LLM: Ollama (llama3.1:70b)

🔄 Execution Flow:
   [1/3] Spawning research agent...
   [2/3] Executing research task...
   [3/3] Processing results...

✅ Results:
    Research Summary:
    - AI agents are increasingly using multi-agent architectures
    - LLM orchestration frameworks are gaining adoption
    - Key trends: tool use, planning, memory systems
    - Enterprise focus: governance, observability, safety

    Sources: 3 web pages analyzed
    Confidence: 0.85
```

### Step 6: Create Your First Workflow

```bash
# Create a new workflow
kautilya manifest new

# Follow the interactive prompts:
# ? Manifest name: my-workflow
# ? Description: My first agent workflow
# ? Add step: research
#   ? Role: research
#   ? Capabilities: web_search, summarize
# ? Add step: synthesize
#   ? Role: synthesis
#   ? Capabilities: none
```

This generates `manifests/my-workflow.yaml`:

```yaml
manifest_id: my-workflow
name: My First Agent Workflow
version: "1.0.0"

steps:
  - id: research
    role: research
    capabilities: [web_search, summarize]
    inputs:
      - name: query
        source: user_input
    outputs: [research_snippet]
    timeout: 30

  - id: synthesize
    role: synthesis
    inputs:
      - name: research
        source: previous_step
    outputs: [final_report]
    timeout: 20
```

**Run the workflow:**
```bash
kautilya manifest run manifests/my-workflow.yaml \
  --input "Analyze the impact of multi-agent systems on enterprise AI"
```

## 🎮 CLI Usage Guide

The `kautilya` CLI is the primary interface for building and managing agents. It provides an interactive, guided experience for:

- **Agent Creation**: Build agents without writing code
- **Workflow Design**: Create multi-step workflows interactively
- **LLM Configuration**: Switch between 6 providers seamlessly
- **Skill Development**: Scaffold skills with validation
- **MCP Integration**: Add external tools with guided setup
- **Deep Research**: Use as a research platform with multi-agent orchestration

### Quick CLI Examples

**Interactive Mode:**
```bash
# Launch interactive CLI
kautilya

# You'll see:
> _

# Try these commands:
> /agent new my-agent          # Create an agent
> /skill new my-skill          # Create a skill
> /manifest new                # Create a workflow
> /llm config                  # Configure LLM
> /mcp add github              # Add MCP server
> /research --mode interactive # Start research session
```

**Direct Commands:**
```bash
# Create agent in one line
kautilya agent new research-agent --role research --capabilities web_search,summarize

# Run workflow
kautilya manifest run manifests/my-workflow.yaml --input "Your query here"

# Test LLM connection
kautilya llm test

# List available skills
kautilya skill list
```

### Interactive Research Platform

Use `kautilya` as a deep research platform:

```bash
> /research --mode interactive

research> /query "Latest trends in multi-agent AI systems"

# Multi-phase research:
# 1. Initial search (web, academic, code repos)
# 2. Confidence-based deep dive
# 3. Fact verification
# 4. Progressive report building

research> /refine "game-theoretic approaches"
research> /verify
research> /sources
research> /export markdown
```

### Complete Guide

📖 **[Read the full CLI Usage Guide](docs/cli-usage.md)** for:
- 10 complete interactive sessions (beginner → advanced)
- Deep research platform tutorial (42-minute example)
- Parallel workflow orchestration
- Complete command reference (40+ commands)
- Tips, tricks, and troubleshooting

## 🎭 Multi-Agent Patterns

The framework supports multiple orchestration patterns:

### 1. **Pipeline (Sequential)**
Agents execute in a linear sequence:
```
User → Research Agent → Verification Agent → Synthesis Agent → Output
```
**Use cases**: Research reports, data analysis, content generation

### 2. **Hierarchical Delegation**
Lead agent delegates subtasks to specialized agents:
```
         Lead Agent
        /     |     \
   Research  Code  Analysis
```
**Use cases**: Complex problem decomposition, project planning

### 3. **Fan-Out/Fan-In (Parallel)**
Multiple agents work in parallel, results merged:
```
        Lead Agent
       /    |    \
   Agent1 Agent2 Agent3
       \    |    /
      Aggregator Agent
```
**Use cases**: Distributed search, parallel validation, consensus building

### 4. **Supervisor Pattern**
Supervisor monitors and validates agent outputs:
```
Worker Agent → Supervisor Agent → (retry or approve)
```
**Use cases**: Quality control, compliance checking, safety validation

### 5. **Specialist Team**
Domain-specific agents collaborate:
```
Product Manager Agent ↔ Engineering Agent ↔ QA Agent
```
**Use cases**: Software development, cross-functional workflows

## 🌐 LLM Provider Support

### Supported Providers (6)

| Provider | Type | Models | Best For | Cost | Status |
|----------|------|--------|----------|------|--------|
| **Ollama** ⭐ | Local | Llama 3.1:70b (default), Mistral, Mixtral | Privacy, offline, default | $0 (self-hosted) | **Active** |
| **GitHub Copilot CLI** | Cloud | GPT-4 | Code generation (Ralph) | Included with Copilot | **Active** |
| **OpenAI** | Cloud | GPT-4o, GPT-4o-mini | General purpose | $2.50-15/M tokens | Optional |
| **Anthropic** | Cloud | Claude Opus 4.5, Sonnet 4.5, Haiku 4 | Enterprise, reasoning | $3-15/M tokens | Optional |
| **Azure OpenAI** | Cloud | GPT-4o, GPT-4 Turbo | Enterprise Azure | $2.50-15/M tokens | Optional |
| **Google Gemini** | Cloud | Gemini 2.0 Flash, 1.5 Pro | Multimodal, cost-effective | $0.075-7/M tokens | Optional |
| **vLLM** | Local | Any HuggingFace model | High performance | $0 (self-hosted) | Optional |

### Quick Provider Switch

```bash
# Default: Ollama (no configuration needed)
ollama pull llama3.1:70b

# Switch to cloud provider (optional)
kautilya llm config --provider openai --model gpt-4o

# Or switch at runtime
kautilya manifest run workflow.yaml --llm-provider ollama --llm-model llama3.1:70b
```

## 🎯 Examples

Explore [examples/](examples/) for complete working projects:

| Example | Description | What You'll Learn | Difficulty |
|---------|-------------|-------------------|------------|
| [01-simple-agent](examples/01-simple-agent/) | Single research agent | Basic agent creation, skill binding | ⭐ Beginner |
| 02-multi-step-workflow | Sequential pipeline | Workflow orchestration, artifact passing | ⭐⭐ Intermediate |
| 03-custom-skill | Build your own skill | Skill development, schema validation | ⭐⭐ Intermediate |
| 04-mcp-integration | External tool access | MCP gateway, scoped permissions | ⭐⭐⭐ Advanced |

### Running Examples

```bash
# Navigate to example directory
cd examples/01-simple-agent

# Install any additional dependencies
pip install -r requirements.txt  # if present

# Run the example
python run.py
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/paragajg/agentic-framework.git
cd agentic-framework

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Start infrastructure services
docker-compose up -d

# Verify setup
pytest  # Run tests
```

### Code Quality Standards

```bash
# Format code
black --line-length 100 .

# Type checking
mypy --strict .

# Linting
ruff check .

# Run all checks
make lint  # or manually run all above
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py

# Run specific test
pytest tests/test_orchestrator.py::test_workflow_execution -v
```

### Project Structure

```
agentic-framework/
├── adapters/              # LLM provider adapters (6 providers)
│   ├── llm/
│   │   ├── anthropic.py   # Anthropic Claude
│   │   ├── openai.py      # OpenAI GPT
│   │   ├── azure.py       # Azure OpenAI
│   │   ├── gemini.py      # Google Gemini
│   │   ├── local.py       # Ollama (local)
│   │   └── vllm.py        # vLLM (optimized local)
│   └── tests/
│
├── orchestrator/          # Workflow orchestration engine
│   ├── service/
│   │   ├── workflow_engine.py  # YAML manifest execution
│   │   ├── approvals.py        # Human-in-the-loop
│   │   └── models.py
│   ├── manifests/         # Example workflow YAMLs
│   └── tests/
│
├── subagent-manager/      # Subagent lifecycle management
│   ├── service/
│   │   ├── lifecycle.py        # Spawn, execute, destroy
│   │   ├── governance.py       # RBAC, policies
│   │   ├── validator.py        # Artifact validation
│   │   └── provenance.py       # Audit logging
│   └── tests/
│
├── memory-service/        # Multi-tier memory storage
│   ├── service/
│   │   ├── storage/
│   │   │   ├── redis.py        # Session cache
│   │   │   ├── vector.py       # Milvus/Chroma
│   │   │   ├── postgres.py     # Structured data
│   │   │   └── s3.py           # Cold storage
│   │   ├── embedding.py        # Text embeddings
│   │   └── tasks.py            # Compaction jobs
│   └── tests/
│
├── mcp-gateway/           # MCP tool gateway
│   ├── service/
│   │   ├── catalog.py          # Tool registry
│   │   ├── proxy.py            # Runtime proxy
│   │   ├── auth.py             # Scoped access
│   │   └── rate_limit.py       # Throttling
│   └── tests/
│
├── code-exec/             # Skill executor & sandbox
│   ├── service/
│   │   ├── executor.py         # Sandboxed execution
│   │   ├── skill_registry.py   # Auto-discovery
│   │   └── skill_parser.py     # Dual-format parsing
│   ├── skills/            # Prepackaged skills
│   │   ├── text_summarize/
│   │   ├── extract_entities/
│   │   └── prepackaged/
│   └── tests/
│
├── tools/                 # CLI utilities
│   └── kautilya/          # Developer CLI
│       ├── kautilya/
│       │   ├── commands/       # CLI commands
│       │   ├── memory/         # Memory management
│       │   └── templates/      # Code generation
│       └── tests/
│
├── docs/                  # Documentation
│   ├── schema_registry/   # JSON Schemas for artifacts
│   └── schemas/           # YAML manifest schemas
│
├── examples/              # Example projects
│   ├── 01-simple-agent/
│   ├── 02-multi-step-workflow/
│   ├── 03-custom-skill/
│   └── 04-mcp-integration/
│
├── tests/                 # Integration tests
│   └── integration/
│
├── scripts/               # Utility scripts
├── docker-compose.yml     # Full-stack infrastructure
├── pyproject.toml         # Package metadata
├── requirements.txt       # Dependencies
├── CONTRIBUTING.md        # Contribution guidelines
├── CHANGELOG.md           # Version history
└── LICENSE                # MIT License
```

## 🏢 Production Deployment

### Docker Compose (Development/Testing)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f orchestrator

# Stop services
docker-compose down
```

### Kubernetes (Production)

```bash
# Using Helm charts
helm install agentic-framework ./infra/helm/agentic-framework \
  --set llm.provider=anthropic \
  --set llm.apiKeySecret=anthropic-key \
  --set redis.enabled=true \
  --set postgres.enabled=true \
  --set milvus.enabled=true

# Check deployment
kubectl get pods -n agentic-framework

# Scale orchestrator
kubectl scale deployment orchestrator --replicas=5
```

### Environment Variables

```bash
# LLM Provider (Default: Ollama - no API key needed)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b

# GitHub Copilot CLI (for Ralph loop)
GITHUB_TOKEN=ghp_...

# Optional Cloud Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
GEMINI_API_KEY=...

# Infrastructure
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/agentic
MILVUS_URL=http://localhost:19530
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Services
ORCHESTRATOR_URL=http://localhost:8000
SUBAGENT_MANAGER_URL=http://localhost:8001
MEMORY_SERVICE_URL=http://localhost:8002
MCP_GATEWAY_URL=http://localhost:8003
CODE_EXEC_URL=http://localhost:8004

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# Security
JWT_SECRET_KEY=your-secret-key
ENABLE_RBAC=true
REQUIRE_APPROVALS=true
```

### Performance Tuning

```yaml
# orchestrator/config.yaml
performance:
  max_concurrent_workflows: 100
  subagent_pool_size: 50
  connection_pool_size: 20
  cache_ttl: 3600

memory:
  compaction_interval: 300  # seconds
  max_session_size_mb: 100
  vector_batch_size: 1000

timeouts:
  default_step_timeout: 30
  max_workflow_timeout: 3600
  llm_request_timeout: 60
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make** your changes:
   - Follow code quality standards (Black, mypy, ruff)
   - Add tests (maintain 90%+ coverage)
   - Update documentation
4. **Test** your changes:
   ```bash
   pytest
   black --check .
   mypy --strict .
   ```
5. **Commit** with conventional commits:
   ```bash
   git commit -m "feat: add amazing feature"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open** a Pull Request

### Development Workflow

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes and test
pytest
black --line-length 100 .
mypy --strict .

# Commit with conventional commits
git commit -m "feat: add new skill registry feature"

# Push and create PR
git push origin feature/my-feature
gh pr create --title "feat: add new skill registry feature"
```

## ❓ FAQ

### General Questions

**Q: What makes this different from LangChain or AutoGen?**
A: We focus on production-grade features: typed artifacts with provenance, LLM-agnostic architecture, declarative YAML workflows, enterprise security (RBAC, audit trails), and multi-tier memory management. LangChain/AutoGen are great for prototyping but lack governance features for production.

**Q: Can I use local/open-source LLMs?**
A: Yes! We support Ollama (easiest) and vLLM (fastest) for running models like Llama 3.1, Mistral, and any HuggingFace model locally with zero API costs.

**Q: Is this production-ready?**
A: Yes. The framework includes comprehensive tests (90%+ coverage), observability (OpenTelemetry), security (RBAC, audit trails), and has been designed for enterprise workloads. We recommend starting with pilot projects.

**Q: How does pricing work?**
A: The framework itself is MIT-licensed (free). You pay only for:
- LLM provider API calls (Anthropic/OpenAI/Azure/Gemini) OR
- Infrastructure costs for local LLMs (compute, storage)
- Cloud infrastructure (if deployed on AWS/GCP/Azure)

### Technical Questions

**Q: How do I switch LLM providers?**
A: Use `kautilya llm config --provider <name>` or set environment variables. No code changes needed - all adapters use the same interface.

**Q: What's the difference between Skills and MCP Tools?**
A: **Skills** are deterministic Python functions you write (e.g., data processing, calculations). **MCP Tools** are external services accessed via Model Context Protocol (e.g., GitHub API, file system, databases).

**Q: How does memory compaction work?**
A: The Memory Service automatically summarizes and archives old conversation data based on configurable rules (time-based, token-based, or custom). This prevents context window overflow while preserving important information.

**Q: Can agents call other agents?**
A: Yes! Use the **Hierarchical Delegation** pattern where a lead agent spawns subagents. Each subagent has isolated context and capabilities.

**Q: How do I add a new LLM provider?**
A: Implement the `BaseLLMAdapter` interface in `adapters/llm/`. See existing adapters for examples. Contributions welcome!

**Q: What's the maximum workflow size?**
A: No hard limit. We've tested workflows with 50+ steps. Performance depends on your infrastructure (Redis, Postgres, etc.) and LLM provider rate limits.

### Security Questions

**Q: How are skills sandboxed?**
A: Skills execute in isolated Python processes with restricted imports and filesystem access. You can configure safety flags (`file_system`, `network_access`) to control permissions.

**Q: How is PII handled?**
A: The MCP Gateway includes configurable PII filters. You can also mark skills with `pii_risk` flag to require approval before execution.

**Q: What audit/compliance features exist?**
A: Every artifact includes full provenance (who, what, when, why, inputs, outputs). All actions logged to Postgres for compliance auditing. RBAC controls who can create/execute workflows.

### Deployment Questions

**Q: What are the minimum infrastructure requirements?**
A: For development: Docker Desktop. For production: Kubernetes cluster with 4GB RAM (orchestrator + services) + LLM provider or local GPU for inference.

**Q: Can I run this without Docker?**
A: Yes, but you'll need to manually install and configure Redis, Postgres, and Milvus. See `docker-compose.yml` for service configurations.

**Q: How do I monitor production deployments?**
A: Use the included Prometheus metrics and Grafana dashboards. All services expose `/metrics` endpoints with OpenTelemetry instrumentation.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [anyio](https://anyio.readthedocs.io/) - Async I/O
- [SQLModel](https://sqlmodel.tiangolo.com/) - SQL databases
- [Redis](https://redis.io/) - In-memory cache
- [PostgreSQL](https://www.postgresql.org/) - Relational database
- [Milvus](https://milvus.io/) / [Chroma](https://www.trychroma.com/) - Vector databases
- [MinIO](https://min.io/) - S3-compatible object storage

LLM Providers:
- [Anthropic Claude](https://www.anthropic.com/)
- [OpenAI GPT](https://openai.com/)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Google Gemini](https://ai.google.dev/)
- [Ollama](https://ollama.ai/)
- [vLLM](https://github.com/vllm-project/vllm)

Standards:
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [OpenTelemetry](https://opentelemetry.io/)
- [JSON Schema](https://json-schema.org/)

## 📞 Support

### Documentation

- **[CLI Usage Guide](docs/cli-usage.md)** - Interactive CLI tutorial (beginner → advanced)
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Workflow Manifests](docs/manifests.md)** - YAML workflow guide
- **[Skills Development](docs/skills.md)** - Creating custom skills
- **[MCP Integration](docs/mcp.md)** - External tool integration
- **[Examples](examples/)** - Working code examples

### Community & Support

- **Issues**: [GitHub Issues](https://github.com/paragajg/agentic-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paragajg/agentic-framework/discussions)
- **Email**: dev@agentic-framework.org

## 🗺️ Roadmap

See [CHANGELOG.md](CHANGELOG.md) for version history and [GitHub Projects](https://github.com/paragajg/agentic-framework/projects) for upcoming features.

### Upcoming Features (v1.1.0)
- [ ] Web UI for workflow visualization
- [ ] GraphQL API support
- [ ] Streaming responses
- [ ] Advanced routing patterns (conditional, loop)
- [ ] Multi-tenancy support
- [ ] Cost tracking and budget alerts

### Future Releases
- [ ] Distributed execution (Celery/Prefect)
- [ ] Fine-tuned models for specific tasks
- [ ] Browser-based agent playground
- [ ] Mobile SDK (iOS/Android)
- [ ] More MCP servers (Salesforce, Notion, etc.)

---

**Made with ❤️ by the Agentic Framework community**

⭐ **Star this repo** if you find it useful!
🐛 **Report bugs** via [GitHub Issues](https://github.com/paragajg/agentic-framework/issues)
💡 **Suggest features** via [Discussions](https://github.com/paragajg/agentic-framework/discussions)
