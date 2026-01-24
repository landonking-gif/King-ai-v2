# Ralph Code Agent Integration

## Overview

The Ralph Code Agent is a specialized subagent in King AI v3 that delegates code implementation tasks to the Ralph autonomous coding loop running on your AWS server.

## Architecture

```
User Request → Orchestrator → Ralph Code Agent
                                     ↓
                            Generate PRD
                                     ↓
                            SSH to AWS Server
                                     ↓
                            Execute Ralph Loop
                                     ↓
                            Collect Results
                                     ↓
                            Return to Orchestrator
```

## Features

- **Automated PRD Generation**: Creates detailed Product Requirements Documents from high-level requests
- **AWS Integration**: Executes Ralph on your AWS server (54.167.201.176)
- **Human Approval Gates**: Optional approval before code execution
- **Full Provenance**: Tracks all changes with audit trail
- **File Monitoring**: Reports all files created or modified
- **Error Handling**: Comprehensive timeout and error management

## Usage

### Via Dashboard (Talk to King AI)

Simply request code implementation tasks:

```
"Implement a new REST API endpoint for user authentication"

"Create a database migration script to add email verification"

"Refactor the payment processing module to support multiple currencies"
```

The orchestrator will automatically:
1. Detect it's a code task
2. Generate a detailed PRD
3. Request your approval
4. Execute Ralph on AWS
5. Return results with full provenance

### Via CLI (kautilya)

```bash
# Run the Ralph code implementation workflow
kautilya manifest run orchestrator/manifests/ralph-code-implementation.yaml \
  --input '{
    "task_description": "Add user authentication API",
    "requirements": [
      "JWT token-based authentication",
      "Login and logout endpoints",
      "Password hashing with bcrypt",
      "Rate limiting on login attempts"
    ],
    "files_context": [
      "src/api/auth.py",
      "src/models/user.py"
    ]
  }'
```

### Via API

```python
import requests

response = requests.post("https://king-ai-studio.me/api/workflows/execute", json={
    "manifest_id": "ralph-code-implementation",
    "inputs": {
        "task_description": "Implement user authentication",
        "requirements": [
            "JWT tokens",
            "Login/logout endpoints",
            "Password hashing"
        ],
        "target_server": "54.167.201.176"
    }
})

print(response.json())
```

## Configuration

### Environment Variables

Add to your `.env` files:

```bash
# Orchestrator .env
RALPH_ENABLED=true
RALPH_DEFAULT_SERVER=54.167.201.176
RALPH_SSH_KEY_PATH=/path/to/king-ai-studio.pem
RALPH_APPROVAL_REQUIRED=true
RALPH_EXECUTION_TIMEOUT=900
```

### Workflow Configuration

Edit `orchestrator/manifests/ralph-code-implementation.yaml`:

- **Approval Gate**: Set `approval_gates.approval_type` to `none` to disable approval
- **Timeout**: Adjust `steps[1].timeout` for longer/shorter execution time
- **Target Server**: Change `inputs.target_server.default` for different servers

## PRD Format

The Ralph agent expects a structured PRD:

```json
{
  "title": "Task title",
  "description": "Detailed description of what to implement",
  "requirements": [
    "Specific requirement 1",
    "Specific requirement 2"
  ],
  "files_to_modify": [
    "path/to/file1.py",
    "path/to/file2.py"
  ],
  "acceptance_criteria": [
    "Criterion 1",
    "Criterion 2"
  ],
  "context": "Additional context about the codebase"
}
```

## Example Workflows

### 1. New Feature Implementation

```yaml
task_description: "Add email notification system"
requirements:
  - "Send welcome emails on user registration"
  - "Send password reset emails"
  - "Use SendGrid API"
  - "Template-based emails with Jinja2"
files_context:
  - "src/services/email.py"
  - "templates/emails/"
```

### 2. Bug Fix

```yaml
task_description: "Fix memory leak in data processing pipeline"
requirements:
  - "Identify and fix memory leak in workers"
  - "Add proper resource cleanup"
  - "Add memory usage monitoring"
files_context:
  - "src/workers/processor.py"
  - "src/utils/resources.py"
```

### 3. Refactoring

```yaml
task_description: "Refactor database queries to use SQLAlchemy"
requirements:
  - "Replace raw SQL with SQLAlchemy ORM"
  - "Maintain existing functionality"
  - "Add proper error handling"
  - "Update tests"
files_context:
  - "src/database/queries.py"
  - "tests/test_database.py"
```

## Monitoring

### View Ralph Execution Logs

```bash
# On AWS server
ssh -i king-ai-studio.pem ubuntu@54.167.201.176
tail -f /var/log/ralph/execution.log
```

### Check Workflow Status

```bash
# Via CLI
kautilya workflow status <workflow_id>

# Via API
curl https://king-ai-studio.me/api/workflows/<workflow_id>/status
```

### View Provenance

All Ralph executions are tracked in the Memory Service:

```python
from memory_service import get_provenance

provenance = get_provenance(artifact_id="ralph-execution-123")
print(provenance.actor_id)  # "ralph-code-agent"
print(provenance.files_changed)
print(provenance.timestamp)
```

## Security

### Approval Gates

By default, Ralph execution requires human approval:

1. PRD is generated and presented
2. Human reviews and approves/rejects
3. If approved, Ralph executes on AWS
4. Results are validated and returned

### SSH Security

- Uses SSH key authentication (king-ai-studio.pem)
- No password authentication
- StrictHostKeyChecking disabled for automated workflows
- Keys should have 600 permissions

### Sandboxing

Ralph executes in an isolated environment on AWS:
- Limited file system access
- Network access controlled by AWS security groups
- Execution timeout prevents runaway processes
- All changes tracked with provenance

## Troubleshooting

### Common Issues

**Issue**: "SSH key not found"
```bash
# Solution: Verify key path
export RALPH_SSH_KEY_PATH=/full/path/to/king-ai-studio.pem
chmod 600 $RALPH_SSH_KEY_PATH
```

**Issue**: "Ralph execution timed out"
```yaml
# Solution: Increase timeout in manifest
steps:
  - id: ralph_execution
    timeout: 1800  # 30 minutes
```

**Issue**: "Failed to upload PRD to server"
```bash
# Solution: Check AWS security group allows SSH (port 22)
# Verify SSH key permissions
ls -la king-ai-studio.pem  # Should be -rw-------
```

**Issue**: "Ralph not found on server"
```bash
# Solution: Verify Ralph installation on AWS
ssh -i king-ai-studio.pem ubuntu@54.167.201.176 \
  "which python3 && ls -la /home/ubuntu/king-ai-v2/scripts/ralph/"
```

## Advanced Configuration

### Custom Ralph Scripts

To use a different Ralph script:

Edit `handler.py`:
```python
ralph_script_path = "/path/to/custom/ralph_script.py"
```

### Multiple Target Servers

Configure different servers for different tasks:

```python
# In workflow input
{
    "target_server": "production-server.example.com",  # For production
    # or
    "target_server": "staging-server.example.com",     # For staging
    # or
    "target_server": "local"                            # For local testing
}
```

### Parallel Execution

Run multiple Ralph instances in parallel:

```yaml
# Fan-out pattern in manifest
steps:
  - id: ralph_feature_1
    role: code
    capabilities: ["ralph_code_agent"]
    # ...
  
  - id: ralph_feature_2
    role: code
    capabilities: ["ralph_code_agent"]
    # ...
  
  - id: merge_results
    role: synthesis
    # Combine results from both
```

## Integration with Existing Workflows

### Add Ralph to Existing Manifests

```yaml
steps:
  - id: research_phase
    role: research
    # ... existing research step
  
  - id: code_implementation  # Add Ralph step
    role: code
    capabilities: ["ralph_code_agent"]
    skills:
      - name: ralph_code_agent
    inputs:
      - name: prd
        source: previous_step
        key: implementation_plan
  
  - id: testing
    role: verify
    # ... existing verification step
```

## Best Practices

1. **Detailed PRDs**: More detail = better Ralph output
2. **Incremental Tasks**: Break large tasks into smaller PRDs
3. **Test First**: Use staging server before production
4. **Review Changes**: Always review Ralph's output before merging
5. **Monitor Execution**: Check logs during long-running tasks
6. **Version Control**: Ralph should commit to feature branches
7. **Approval Gates**: Keep them enabled for production tasks

## Support

For issues or questions:
- GitHub Issues: [Report a bug](https://github.com/your-repo/issues)
- Documentation: [Full docs](https://king-ai-studio.me/docs)
- Email: support@king-ai-studio.me
