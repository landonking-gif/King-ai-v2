# Kautilya - Enterprise Agentic Framework CLI

Interactive CLI for scaffolding agents, configuring LLMs, creating skills, and managing workflow manifests.

## Features

- 🚀 **Interactive Mode** - REPL-style interface with slash commands
- 🤖 **Agent Scaffolding** - Generate subagents with role templates
- 🛠️ **Skill Creation** - Create deterministic skills with JSON Schema validation
- 🔌 **LLM Configuration** - Multi-provider support (Anthropic, OpenAI, Azure, Local)
- 📡 **MCP Integration** - Manage MCP servers and tool catalogs
- 📋 **Manifest Management** - Create and validate YAML workflow definitions
- 🏃 **Runtime Control** - Start/stop services, view status, tail logs

## Installation

```bash
# Install globally with uv
uv pip install kautilya

# Or install from source
cd tools/kautilya
uv pip install -e .
```

## Quick Start

### Interactive Mode

```bash
$ kautilya
┌─────────────────────────────────────────────────┐
│  Kautilya v1.0 - Agentic Framework CLI          │
│  Type /help for commands, or describe your task │
└─────────────────────────────────────────────────┘

> /help
```

### Command Reference Table

| Command | Description | Options / Actions |
|---------|-------------|-------------------|
| **Project Setup** |||
| `/init` | Initialize new agent project | `--name, --provider, --mcp` |
| **Agent Management** |||
| `/agent new <name>` | Generate new subagent | `--role, --capabilities, --output-type` |
| `/agent list` | List all agents | |
| **Skill Management** |||
| `/skill new <name>` | Scaffold new skill | `--format, --description, --safety-flags` |
| `/skill list` | List all available skills | |
| `/skill import <url>` | Import skill from URL/ZIP | `--format hybrid\|native\|anthropic` |
| `/skill export <name>` | Export skill for sharing | `--output` |
| `/skill validate <name>` | Validate skill format | |
| **LLM Configuration** |||
| `/llm config` | Configure LLM provider | `--provider, --model, --api-key-env` |
| `/llm list` | List available LLM adapters | |
| `/llm test` | Test LLM connection | |
| **MCP Server Management** |||
| `/mcp add <server>` | Add MCP server to manifest | `--scopes, --rate-limit` |
| `/mcp list` | List registered MCP servers | `--all` (include disabled) |
| `/mcp import <file>` | Import MCP server from YAML | |
| `/mcp enable <tool_id>` | Enable a disabled MCP server | |
| `/mcp disable <tool_id>` | Disable an enabled MCP server | |
| `/mcp test <server>` | Test MCP server connection | |
| **Web Search** |||
| `/websearch config` | Configure web search providers | `--provider ddg\|tavily` |
| `/websearch list` | List web search providers | |
| `/websearch test` | Test web search | `<query>` |
| **Manifest Management** |||
| `/manifest new` | Create workflow manifest | `--name, --steps` |
| `/manifest validate` | Validate manifest schema | `<file>` |
| `/manifest run <file>` | Execute workflow | `--dry-run` |
| **Runtime Control** |||
| `/run` | Run project in dev mode | `--detach, --port` |
| `/status` | Show service status | |
| `/logs [agent]` | Tail service logs | `--follow, --lines` |
| **Session Control** |||
| `/stats` | Show query statistics | |
| `/verbose` | Toggle verbose mode | `on \| off` |
| `/display [mode]` | Set display mode | `minimal \| detailed \| toggle` |
| `/output [mode]` | Set output verbosity | `concise \| verbose \| toggle` |
| `/chat` | Toggle LLM chat mode | `on \| off` |
| `/clear` | Clear chat history | |
| `/help` | Show help message | |
| `/exit` | Exit interactive mode | (or `/quit`) |

```bash
> /init
? Project name: customer-support-agent
? LLM provider: anthropic
? Enable MCP integration? Yes
✓ Created agent project structure
✓ Generated default configuration
✓ Configured anthropic as LLM provider

> _
```

### Non-Interactive Commands

```bash
# Initialize project
kautilya init --name my-agent --provider anthropic

# Create agent
kautilya agent new research-agent \
  --role research \
  --capabilities web_search,document_read \
  --output-type research_snippet

# Create skill
kautilya skill new extract-entities \
  --description "Extract named entities from text"

# Configure LLM
kautilya llm config --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --api-key-env ANTHROPIC_API_KEY

# Run services
kautilya run

# Check status
kautilya status

# Tail logs
kautilya logs orchestrator
```

## Command Reference

### `/init` - Initialize Project

Create a new agent project with guided setup:

```bash
kautilya init
# Or with options:
kautilya init --name my-agent --provider anthropic --mcp
```

Creates:
```
my-agent/
├── .kautilya/config.yaml
├── agents/
├── skills/
├── manifests/
├── schemas/
└── tests/
```

### `/agent new` - Create Subagent

Generate a new subagent with role-specific capabilities:

```bash
kautilya agent new <name> [--role ROLE] [--capabilities CAPS]
```

**Roles:**
- `research` - Information gathering and synthesis
- `verify` - Fact-checking and validation
- `code` - Code generation and review
- `synthesis` - Content generation and formatting
- `custom` - User-defined role

**Example:**
```bash
kautilya agent new research-agent \
  --role research \
  --capabilities web_search,summarize,extract_entities \
  --output-type research_snippet
```

Generates:
```
agents/research-agent/
├── config.yaml          # Agent configuration
├── capabilities.json    # Capability definitions
└── prompts/system.txt   # System prompt template
```

### `/skill new` - Create Skill

Scaffold a new deterministic skill with JSON Schema:

```bash
kautilya skill new <name> [--description DESC]
```

**Example:**
```bash
kautilya skill new extract-entities \
  --description "Extract named entities from text"
```

Creates:
```
skills/extract_entities/
├── skill.yaml         # Registration metadata
├── schema.json        # I/O JSON Schema
├── handler.py         # Implementation stub
└── test_handler.py    # Test template
```

### `/llm config` - Configure LLM Provider

Set up LLM provider credentials and models:

```bash
kautilya llm config [--provider PROVIDER] [--model MODEL]
```

**Providers:**
- `anthropic` - Claude models (Sonnet, Opus, Haiku)
- `openai` - GPT-4o, GPT-4o-mini
- `azure` - Azure OpenAI
- `local` - Local models (Ollama, vLLM)

**Example:**
```bash
kautilya llm config \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --api-key-env ANTHROPIC_API_KEY
```

Generates `.kautilya/llm.yaml`:
```yaml
providers:
  anthropic:
    default_model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    fallback_model: claude-haiku-4-20250514
default_provider: anthropic
```

### `/mcp add` - Add MCP Server

Add MCP server to project manifests:

```bash
kautilya mcp add <server> [--scopes SCOPES] [--rate-limit N]
```

**Example:**
```bash
kautilya mcp add github \
  --scopes "repo:read,issues:write" \
  --rate-limit 60
```

### `/manifest new` - Create Workflow

Create a new workflow manifest with guided setup:

```bash
kautilya manifest new
```

Interactive prompts for:
- Manifest name and description
- Workflow steps (role, agent, capabilities)
- Memory configuration (compaction, token budget)
- Policy settings (approval, tool restrictions)

Generates `manifests/<name>.yaml`:
```yaml
manifest_id: customer-support-workflow
name: Customer Support Workflow
version: 1.0.0
steps:
  - id: step-1
    role: research
    agent: research-agent
    capabilities: [ticket_read, kb_search]
    timeout: 30
memory:
  persist_on: [on_complete]
  compaction:
    strategy: summarize
    max_tokens: 8000
tools:
  catalog_ids: []
policies:
  requires_human_approval: false
```

### `/run` - Run Services

Start all agent framework services using docker-compose:

```bash
kautilya run
```

Starts:
- Orchestrator (port 8000)
- Subagent Manager (port 8001)
- Memory Service (port 8002)
- MCP Gateway (port 8080)
- Code Executor (port 8004)

### `/status` - Check Status

Show status of all running services:

```bash
kautilya status
```

Output:
```
┌─────────────────┬───────────────┬─────────────────────────┐
│ Service         │ Status        │ URL                     │
├─────────────────┼───────────────┼─────────────────────────┤
│ Orchestrator    │ ✓ Running     │ http://localhost:8000   │
│ Subagent Mgr    │ ✓ Running     │ http://localhost:8001   │
│ Memory Service  │ ✓ Running     │ http://localhost:8002   │
│ MCP Gateway     │ ✓ Running     │ http://localhost:8080   │
│ Code Executor   │ ✓ Running     │ http://localhost:8004   │
└─────────────────┴───────────────┴─────────────────────────┘
```

### `/logs` - Tail Logs

Stream logs from services:

```bash
# All services
kautilya logs

# Specific service
kautilya logs orchestrator
kautilya logs memory
kautilya logs mcp
```

## Configuration

### Project Configuration (`.kautilya/config.yaml`)

```yaml
project:
  name: my-agent
  version: 1.0.0

defaults:
  llm_provider: anthropic
  memory_backend: redis
  vector_db: chroma

mcp_gateway:
  url: http://localhost:8080
  auth: bearer_token

orchestrator:
  url: http://localhost:8000
```

### LLM Configuration (`.kautilya/llm.yaml`)

```yaml
providers:
  anthropic:
    default_model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    fallback_model: claude-haiku-4-20250514

  openai:
    default_model: gpt-4o
    api_key_env: OPENAI_API_KEY

  local:
    endpoint: http://localhost:11434
    model: llama3.1:70b

default_provider: anthropic
```

## Environment Variables

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
AZURE_OPENAI_KEY=...

# Infrastructure
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:password@localhost:5432/agentic_framework
VECTOR_DB_URL=http://localhost:19530
MCP_GATEWAY_URL=http://localhost:8080

# Kautilya
KAUTILYA_CONFIG=.kautilya/
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy kautilya/

# Format code
black --line-length 100 kautilya/

# Lint
ruff check kautilya/
```

## Architecture

Kautilya integrates with the Enterprise Agentic Framework services:

```
Kautilya (CLI)
    ↓
Orchestrator (8000) ← Manages workflows
    ↓
Subagent Manager (8001) ← Spawns isolated agents
    ↓
Memory Service (8002) ← Stores artifacts & provenance
    ↓
MCP Gateway (8080) ← Tool catalog & proxy
    ↓
Code Executor (8004) ← Executes deterministic skills
```

## Team Standards

- **Python 3.11+** required
- **Type hints** on all functions
- **Black formatting** (100-char line length)
- **Pydantic** for validation
- **Async/await** with anyio for concurrency

## License

MIT
