# Ralph: Autonomous AI Agent Loop

Ralph is an autonomous AI coding assistant that iteratively implements Product Requirement Documents (PRDs) using GitHub Copilot CLI. Based on Geoffrey Huntley's Ralph pattern, it maintains fresh context across iterations while accumulating learnings.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Usage](#usage)
- [PRD Format](#prd-format)
- [Quality Assurance](#quality-assurance)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)

## Overview

Ralph automates the implementation of software features by:
- Breaking down PRDs into small, implementable user stories
- Generating detailed prompts for each story
- Using GitHub Copilot CLI to implement code changes
- Running quality checks to ensure code standards
- Committing successful implementations
- Accumulating learnings across iterations

### Key Benefits
- **Autonomous Execution**: Runs without human intervention per story
- **Fresh Context**: Each iteration starts clean, avoiding context limits
- **Quality Gates**: Automated testing prevents bad code commits
- **Memory Persistence**: Learns from past iterations
- **Project Awareness**: Adapts to your codebase conventions

## Architecture

```
PRD JSON ──┐
           ├── Ralph Loop ──┤
Progress.txt─┘             ├── Copilot CLI ─── Quality Checks ─── Commit
Git History ───────────────┘
```

### Components
- **ralph.sh**: Main bash script orchestrating the loop
- **prompt.md**: Template for Copilot prompts
- **prd.json**: User stories with completion status
- **progress.txt**: Learnings and context from iterations
- **Skills**: Copilot skills for PRD generation and conversion

## Prerequisites

### Required
- **GitHub Copilot Subscription**: Active Copilot plan
- **GitHub Copilot CLI**: Installed and authenticated
- **Node.js**: For Copilot CLI (v16+)
- **Python 3.8+**: For JSON processing
- **Git**: Version control
- **Bash**: Shell environment

### Optional
- **jq**: Alternative JSON processor (script uses Python by default)
- **Quality Tools**: pytest, ruff, mypy, bandit (configured in prompt.md)

## Installation

### 1. Install Dependencies
```bash
# GitHub Copilot CLI
winget install GitHub.Copilot

# Node.js (if not already installed)
winget install OpenJS.NodeJS

# Authenticate Copilot
copilot
/login  # Follow prompts
```

### 2. Install Ralph
Ralph is already installed in your project at `scripts/ralph/`.

### 3. Configure Skills (Optional)
Skills are pre-installed in:
- `.copilot/skills/prd/`
- `.copilot/skills/ralph/`
- `.opencode/skill/prd/`
- `.opencode/skill/ralph/`

## Workflow

### 1. Create PRD
Use the PRD skill to generate requirements:
```
/load prd
Create a PRD for [feature description]
```

This saves to `tasks/prd-[feature-name].md`

### 2. Convert to Ralph Format
Use the Ralph skill to convert markdown to JSON:
```
/load ralph
Convert tasks/prd-[feature-name].md to prd.json
```

### 3. Run Auto Ralph (Fully Automated)
For complete automation from feature description to implementation:
```bash
./scripts/ralph/auto_ralph.sh "Add user authentication system with login, logout, and password reset"
```

This will:
- Generate a PRD using Copilot
- Convert it to JSON format
- Run Ralph until all stories are completed

### 4. Monitor Progress
- Check `progress.txt` for learnings
- Review git log for commits
- Monitor `prd.json` for completion status

## Configuration

### Customizing Prompts
Edit `scripts/ralph/prompt.md` to include:
- Project-specific conventions
- Additional quality checks
- Codebase patterns
- Team preferences

### Quality Checks
Modify the quality check section in `ralph.sh`:
```bash
# Add your checks here
pytest
ruff check
mypy
bandit
```

### Project Context
Update the prompt template with:
- Architecture details
- Framework preferences
- Coding standards
- Common gotchas

## Usage

### Basic Usage
```bash
# Run with default 10 iterations
./scripts/ralph/ralph.sh

# Run specific number of iterations
./scripts/ralph/ralph.sh 5

# Run until completion
./scripts/ralph/ralph.sh 100
```

### Monitoring
```bash
# Check current status
cat prd.json | python3 -c "import json, sys; data=json.load(sys.stdin); print(f\"Completed: {sum(1 for s in data['userStories'] if s['passes'])}/{len(data['userStories'])}\")"

# View recent progress
tail -20 progress.txt

# Check git history
git log --oneline -10
```

### Manual Intervention
If needed, you can:
- Edit `prd.json` to modify stories
- Update `progress.txt` with additional context
- Manually commit changes
- Skip problematic stories by marking as `passes: true`

## PRD Format

### JSON Structure
```json
{
  "branchName": "feature/my-feature",
  "userStories": [
    {
      "id": "story-1",
      "title": "Add user authentication",
      "description": "Implement login/logout functionality",
      "acceptanceCriteria": "Users can register, login, logout. Passwords hashed.",
      "passes": false,
      "priority": 1
    }
  ]
}
```

### Story Guidelines
- **Small Scope**: Each story should be implementable in one context window
- **Clear Acceptance**: Specific, testable criteria
- **Independent**: Minimal dependencies on other stories
- **Valuable**: Provides business value when complete

### Conversion from Markdown
The Ralph skill converts markdown PRDs with this structure:
```markdown
# Feature: User Authentication

## User Story 1: Login Form
**As a** user
**I want** to login
**So that** I can access my account

**Acceptance Criteria:**
- Form validates email/password
- Shows error messages
- Redirects on success
```

## Quality Assurance

### Automated Checks
Ralph runs these checks after each implementation:
- **pytest**: Unit and integration tests
- **ruff**: Code linting and formatting
- **mypy**: Type checking
- **bandit**: Security scanning

### Manual Verification
For UI stories, include in acceptance criteria:
- "Verify in browser using dev-tools"
- Ralph will prompt for manual verification

### Failure Handling
If checks fail:
- Changes are not committed
- Failure is logged to `progress.txt`
- Next iteration continues with next story
- No iteration limit reached, Ralph continues

## Troubleshooting

### Common Issues

#### Copilot Not Launching
```bash
# Check installation
copilot --version

# Re-authenticate
copilot
/login
```

#### JSON Parsing Errors
```bash
# Validate prd.json
python3 -c "import json; json.load(open('prd.json')); print('Valid JSON')"
```

#### Quality Checks Failing
```bash
# Run checks manually
pytest
ruff check
mypy .
bandit -r .
```

#### Permission Issues
```bash
# Make script executable
chmod +x scripts/ralph/ralph.sh
```

### Debug Mode
Add debug output to `ralph.sh`:
```bash
set -x  # Enable debug
./scripts/ralph/ralph.sh 1
```

### Recovery
If Ralph gets stuck:
1. Check `prd.json` for malformed data
2. Review `progress.txt` for error patterns
3. Manually complete problematic stories
4. Restart Ralph

## Advanced Features

### Custom Quality Checks
Add project-specific checks to the script:
```bash
# In ralph.sh, after "# Run quality checks"
npm run lint
docker-compose build
./custom-check.sh
```

### Integration with CI/CD
Ralph can be integrated into CI pipelines:
```yaml
# .github/workflows/ralph.yml
name: Ralph Implementation
on: [push]
jobs:
  ralph:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Ralph
        run: ./scripts/ralph/ralph.sh 1
```

### Multi-Branch Strategy
For complex features:
- Use feature branches per story
- Merge completed stories to main
- Ralph creates `feature/story-{id}` branches

### Learning Enhancement
Add to `AGENTS.md` after each iteration:
- Discovered patterns
- Gotchas encountered
- Useful conventions
- Integration points

### Parallel Execution
For independent stories, run multiple Ralph instances:
```bash
./scripts/ralph/ralph.sh 1 &  # Instance 1
./scripts/ralph/ralph.sh 1 &  # Instance 2
wait
```

## Contributing

### Extending Ralph
- Add new quality checks
- Customize prompts for specific languages/frameworks
- Integrate additional AI assistants
- Add support for different PRD formats

### Skills Development
- Enhance PRD generation prompts
- Improve markdown-to-JSON conversion
- Add validation for PRD completeness

## License

This implementation is based on the Ralph pattern by Geoffrey Huntley and Ryan Carson's work.

## Support

For issues with Ralph:
1. Check this documentation
2. Review `progress.txt` for patterns
3. Test components individually
4. Open issues with debug output

---

**Ralph v1.0** - Autonomous AI Agent Loop for Product Development