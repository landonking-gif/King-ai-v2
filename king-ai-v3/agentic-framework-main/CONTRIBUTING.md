# Contributing to Agentic Framework

Thank you for your interest in contributing to the Agentic Framework! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- uv package manager (recommended) or pip

### Local Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentic-framework.git
   cd agentic-framework
   ```

2. **Create Virtual Environment**
   ```bash
   # Using uv (recommended)
   uv venv --python 3.11
   source .venv/bin/activate

   # Or using standard venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install package in editable mode with dev dependencies
   pip install -e ".[dev]"
   ```

4. **Verify Installation**
   ```bash
   # Check CLI works
   kautilya --version

   # Run tests
   pytest
   ```

## Development Workflow

### Creating a Feature Branch

```bash
# Update your main branch
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-description
```

### Making Changes

1. **Write Code**
   - Follow PEP 8 style guidelines
   - Use type hints for all function signatures
   - Write docstrings for public APIs
   - Keep functions focused and testable

2. **Run Code Quality Checks**
   ```bash
   # Format code with Black
   black --line-length 100 .

   # Type checking with mypy
   mypy --strict .

   # Linting with ruff
   ruff check .
   ```

3. **Write Tests**
   - Add unit tests for new functionality
   - Aim for 90%+ code coverage
   - Use pytest fixtures for common setup
   - Mock external dependencies

4. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=. --cov-report=html

   # Run specific test file
   pytest tests/unit/test_orchestrator.py

   # Run with verbose output
   pytest -v
   ```

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(orchestrator): add parallel workflow execution"
git commit -m "fix(subagent): resolve memory leak in context cleanup"
git commit -m "docs(readme): update installation instructions"
```

### Submitting a Pull Request

1. **Push Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

3. **PR Requirements**
   - All tests must pass
   - Code coverage should not decrease
   - No linting errors
   - Documentation updated (if applicable)
   - CHANGELOG.md updated (for significant changes)

4. **Code Review Process**
   - Address reviewer feedback
   - Push additional commits to your branch
   - PR will be merged once approved

## Code Standards

### Python Style

- **PEP 8**: Follow Python's style guide
- **Line Length**: 100 characters max
- **Type Hints**: All public functions must have type hints
- **Docstrings**: Use Google-style docstrings

Example:
```python
from typing import Dict, Any, Optional

def process_artifact(
    artifact_data: Dict[str, Any],
    validate: bool = True
) -> Optional[str]:
    """
    Process and validate an artifact.

    Args:
        artifact_data: The artifact data to process
        validate: Whether to validate the artifact

    Returns:
        The artifact ID if successful, None otherwise

    Raises:
        ValueError: If artifact_data is invalid
    """
    # Implementation here
    pass
```

### Testing Standards

- **Coverage**: Aim for 90%+ code coverage
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Async Tests**: Use `pytest-asyncio` for async code

Example test:
```python
import pytest
from agentic_framework.orchestrator import WorkflowEngine

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test basic workflow execution."""
    engine = WorkflowEngine()
    manifest = await engine.load_manifest("test-workflow.yaml")

    result = await engine.execute_workflow(
        manifest,
        user_input={"query": "test"}
    )

    assert result.status == "completed"
    assert len(result.step_artifacts) > 0
```

### Documentation Standards

- **Code Comments**: Explain "why", not "what"
- **Docstrings**: Document all public APIs
- **README**: Keep installation instructions up to date
- **Examples**: Provide working code examples

## Project Structure

```
agentic-framework/
├── adapters/              # LLM provider adapters
├── orchestrator/          # Workflow orchestration
├── subagent-manager/      # Subagent lifecycle
├── memory-service/        # Persistence layer
├── mcp-gateway/           # Tool gateway
├── code-exec/             # Skill executor
├── tools/                 # CLI and utilities
├── examples/              # Example projects
├── docs/                  # Documentation
└── tests/                 # Test suite
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed solution (if any)
- Alternative approaches considered

## Getting Help

- **Documentation**: Check [docs/](docs/) directory
- **Examples**: See [examples/](examples/) directory
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

Thank you for contributing to Agentic Framework! 🚀
