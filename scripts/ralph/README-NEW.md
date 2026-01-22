# Ralph: Autonomous AI Agent Loop

Ralph is an autonomous AI coding assistant that iteratively implements Product Requirement Documents (PRDs) using GitHub Copilot CLI. Based on the ["Ralph Wiggum" technique](https://www.humanlayer.dev/blog/brief-history-of-ralph) and [soderlind/ralph](https://github.com/soderlind/ralph).

## Quick Start

```powershell
# First time: Trust the folder in Copilot CLI (one-time setup)
copilot
# Select "2. Yes, and remember this folder" when prompted

# Run a single iteration (test)
.\scripts\ralph\ralph-once.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe

# Run multiple iterations (autonomous loop)
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 10
```

## How It Works

Ralph implements features using the "fresh context" pattern:

1. **Read** - Copilot reads your PRD and progress.txt
2. **Pick** - Chooses the highest priority incomplete item  
3. **Implement** - Writes code for that one feature
4. **Verify** - Runs tests and quality checks
5. **Update** - Marks item complete and logs progress
6. **Commit** - Commits the changes
7. **Repeat** - Until all items pass or signals completion

Each iteration starts with fresh context to avoid context window pollution.

## Prerequisites

### Required
- **GitHub Copilot CLI**: Version 0.0.387+
  ```powershell
  npm i -g @github/copilot
  copilot --version
  ```
- **Active Copilot Subscription**: GitHub Copilot Pro or Business
- **Folder Trust**: Must trust the folder in Copilot CLI interactively (one-time)
  ```powershell
  copilot  # Then select "Yes, and remember this folder"
  ```
- **Git**: For version control

### Optional
- **Python 3.8+**: For alternative Python-based implementation
- **Quality Tools**: For automated testing (pytest, ruff, mypy)

## Usage

### Single Run (Testing)

```powershell
# Basic usage
.\scripts\ralph\ralph-once.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe

# With custom model
$env:MODEL = "claude-opus-4.5"
.\scripts\ralph\ralph-once.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe

# With skills
.\scripts\ralph\ralph-once.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Skill wp-block-development
```

### Looped Run (Autonomous)

```powershell
# Run for 10 iterations
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 10

# With custom model
$env:MODEL = "gpt-5.2"
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 5

# Run until complete (no iteration limit)
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 999
```

## Configuration

### Permission Profiles

| Profile | Allowed Tools | Use Case |
|---------|--------------|----------|
| `locked` | `write` only | File edits, no shell |
| `safe` | `write`, `git:*`, `npm:*`, `pnpm:*`, `python:*` | Normal dev workflow (recommended) |
| `dev` | All tools except `rm` and `git push` | Broad shell access |

**Always denied**: `shell(rm)`, `shell(git push)`

### Custom Prompts

Create custom prompts in `scripts/ralph/prompts/`:

```powershell
.\scripts\ralph\ralph-once.ps1 -Prompt scripts\ralph\prompts\my-custom-prompt.txt -Prd prd.json -AllowProfile safe
```

### PRD Format

Your `prd.json` should use this format:

```json
[
  {
    "id": "story-1",
    "category": "functional",
    "priority": 1,
    "title": "User Authentication",
    "description": "User can login and logout",
    "steps": ["Navigate to /login", "Enter credentials", "Click login", "Verify redirect"],
    "passes": false
  },
  {
    "id": "story-2",
    "category": "ui",
    "priority": 2,
    "title": "Dashboard UI",
    "description": "User sees dashboard after login",
    "steps": ["Login", "Verify dashboard loads", "Check widgets display"],
    "passes": false
  }
]
```

**Fields:**
- `id`: Unique identifier
- `category`: "functional", "ui", "technical", etc.
- `priority`: Lower number = higher priority
- `title`: Short summary
- `description`: Detailed requirements
- `steps`: Verification steps
- `passes`: `false` (incomplete) → `true` (complete)

## Monitoring Progress

```powershell
# Check progress log
Get-Content progress.txt -Tail 20

# Check git history
git log --oneline -10

# Check PRD status
Get-Content prd.json | ConvertFrom-Json | Where-Object { $_.passes -eq $false }
```

## Troubleshooting

### "Copilot CLI not found"
```powershell
npm i -g @github/copilot
```

### "Folder trust confirmation appears"
Run `copilot` interactively once and select "Yes, and remember this folder"

### "Copilot asks for authentication"
Copilot CLI requires GitHub authentication:
```powershell
# Option 1: GitHub CLI
gh auth login

# Option 2: Use GitHub login in Copilot
copilot
# Follow authentication prompts
```

### "Model not available"
Check available models in Copilot CLI. Common ones:
- `gpt-5.2` (default)
- `claude-opus-4.5`
- `claude-sonnet-4`

### "Permission denied errors"
Adjust the permission profile:
- Use `safe` for normal development
- Use `dev` if you need broader shell access
- Use `locked` for minimal permissions (file edits only)

## Files

- `ralph.ps1` - Looped runner (PowerShell)
- `ralph-once.ps1` - Single-run script (PowerShell)
- `prompts/default.txt` - Default prompt template
- `ralph.py` - Alternative Python implementation (legacy)
- `check_auth.py` - Authentication checker (legacy)

## Advanced Features

### Using Skills

Skills are reusable instruction sets. Create them in `skills/<name>/SKILL.md`:

```powershell
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Skill wp-cli,wp-block-development -Iterations 10
```

### Environment Variables

```powershell
# Set model
$env:MODEL = "claude-opus-4.5"

# Run with custom settings
.\scripts\ralph\ralph.ps1 -Prompt scripts\ralph\prompts\default.txt -Prd prd.json -AllowProfile safe -Iterations 10
```

## Best Practices

1. **Start Small**: Test with `ralph-once.ps1` before running loops
2. **Trust Folder**: Always trust the folder on first run to avoid prompts
3. **Use Safe Profile**: Start with `safe` profile for security
4. **Monitor Progress**: Check `progress.txt` regularly
5. **Review Commits**: Inspect git commits after each iteration
6. **Small Stories**: Break large features into small, testable stories
7. **Clear Acceptance**: Define clear `steps` in PRD for verification

## References

- [Ralph Wiggum Technique](https://www.humanlayer.dev/blog/brief-history-of-ralph)
- [Original soderlind/ralph](https://github.com/soderlind/ralph)
- [Ship code while you sleep (video)](https://www.youtube.com/watch?v=_IK18goX4X8)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

## License

MIT
