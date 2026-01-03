# Agentic Framework Examples

This directory contains example projects demonstrating various features of the Agentic Framework.

## Examples Overview

### 01-simple-agent/
**Difficulty**: Beginner
**Concepts**: Basic agent creation, skill binding, LLM integration

Create and run a single research agent with skills and tools.

```bash
cd 01-simple-agent
python run.py
```

### 02-multi-step-workflow/ (Coming Soon)
**Difficulty**: Intermediate
**Concepts**: Workflow orchestration, multiple agents, typed artifacts

Orchestrate multiple agents in a sequential workflow with artifact passing.

### 03-custom-skill/ (Coming Soon)
**Difficulty**: Intermediate
**Concepts**: Skill development, handler functions, schema validation

Build a custom deterministic skill from scratch.

### 04-mcp-integration/ (Coming Soon)
**Difficulty**: Advanced
**Concepts**: MCP tools, gateway configuration, external services

Integrate external tools using the MCP (Model Context Protocol) gateway.

## Running Examples

### Prerequisites

1. **Install Framework**
   ```bash
   # From repository root
   pip install -e .
   ```

2. **Set API Keys**
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   # or
   export OPENAI_API_KEY="your-key"
   ```

3. **Start Required Services** (for advanced examples)
   ```bash
   # From repository root
   docker-compose up -d
   ```

### Running an Example

```bash
cd examples/01-simple-agent
python run.py
```

## Example Structure

Each example follows this structure:

```
example-name/
├── README.md           # Example documentation
├── run.py             # Main script
├── config.yaml        # Configuration (if applicable)
└── agents/            # Agent definitions (if applicable)
```

## Learning Path

1. **Start Here**: `01-simple-agent/` - Understand basic concepts
2. **Next**: `02-multi-step-workflow/` - Learn orchestration
3. **Then**: `03-custom-skill/` - Extend functionality
4. **Advanced**: `04-mcp-integration/` - External integrations

## Getting Help

- **Documentation**: See `../docs/` directory
- **Issues**: Report bugs at https://github.com/your-org/agentic-framework/issues
- **Questions**: Use GitHub Discussions

## Contributing Examples

Have a great example to share? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on submitting examples.

### Example Contribution Guidelines

- Keep examples simple and focused
- Include comprehensive README
- Test before submitting
- Add comments explaining key concepts
- Use realistic use cases
