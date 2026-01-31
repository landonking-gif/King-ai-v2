# Ralph Code Agent - Deployment Guide

This guide walks through deploying the Ralph Code Agent skill to your King AI v3 orchestrator.

## Prerequisites

✅ King AI v3 services running (Orchestrator, Subagent Manager, Code Executor, MCP Gateway, Memory Service)  
✅ AWS EC2 instance with Ralph loop installed  
✅ SSH key for AWS access (`king-ai-studio.pem`)  
✅ Ollama with llama3.1:70b model  
✅ PostgreSQL database for provenance tracking  

## Step 1: Configure Environment

Add these environment variables to your orchestrator configuration:

```bash
# .env or docker-compose.yml
RALPH_SERVER_HOST=54.167.201.176
RALPH_SSH_KEY_PATH=/home/ubuntu/.ssh/king-ai-studio.pem
RALPH_SSH_USER=ubuntu
RALPH_EXECUTION_TIMEOUT=600
RALPH_SCRIPTS_PATH=/home/ubuntu/king-ai-v2/scripts/ralph
```

## Step 2: Deploy Skill Files

Copy the Ralph Code Agent skill to your Code Executor service:

```bash
# From your local machine
cd king-ai-v3/agentic-framework-main/code-exec

# Sync to AWS server
rsync -avz -e "ssh -i king-ai-studio.pem" \
  skills/ralph_code_agent/ \
  ubuntu@54.167.201.176:/home/ubuntu/king-ai-v3/agentic-framework-main/code-exec/skills/ralph_code_agent/
```

## Step 3: Register Skill with Code Executor

The skill should be automatically discovered by the Code Executor service on restart. Verify registration:

```bash
# SSH to AWS server
ssh -i king-ai-studio.pem ubuntu@54.167.201.176

# Check skill registry
curl http://localhost:8003/skills | jq '.skills[] | select(.name=="ralph_code_agent")'
```

Expected output:
```json
{
  "name": "ralph_code_agent",
  "version": "1.0.0",
  "description": "Delegates code implementation to Ralph autonomous loop",
  "parameters": {
    "prd": "object",
    "target_server": "string",
    "approve_before_execution": "boolean"
  },
  "safety_flags": ["file_system", "network_access", "side_effect"]
}
```

## Step 4: Deploy Workflow Manifest

Copy the workflow manifest to the orchestrator:

```bash
# From your local machine
rsync -avz -e "ssh -i king-ai-studio.pem" \
  orchestrator/manifests/ralph-code-implementation.yaml \
  ubuntu@54.167.201.176:/home/ubuntu/king-ai-v3/agentic-framework-main/orchestrator/manifests/
```

Verify manifest registration:

```bash
# SSH to AWS
ssh -i king-ai-studio.pem ubuntu@54.167.201.176

# Check workflow registry
curl http://localhost:8000/workflows | jq '.workflows[] | select(.name=="ralph-code-implementation")'
```

## Step 5: Test SSH Connectivity

Verify the orchestrator can connect to Ralph on AWS:

```bash
# SSH to AWS
ssh -i king-ai-studio.pem ubuntu@54.167.201.176

# Test SSH from orchestrator container to localhost (Ralph location)
docker exec -it orchestrator bash

# Inside container
ssh -i /home/ubuntu/.ssh/king-ai-studio.pem ubuntu@localhost 'echo "SSH works!"'
```

**Troubleshooting:**
- Ensure SSH key is mounted in orchestrator container
- Verify key permissions: `chmod 600 king-ai-studio.pem`
- Check AWS security group allows SSH (port 22) from orchestrator

## Step 6: Run Test Execution

Test the Ralph Code Agent with a simple task:

```bash
# On AWS server
cd /home/ubuntu/king-ai-v3/agentic-framework-main/code-exec/skills/ralph_code_agent

# Run test suite
python test_handler.py
```

Expected output:
```
🧪 Ralph Code Agent - Unit Tests

================================================================================
Test 2: Approval Flow
================================================================================
...
✅ Approval flow test passed!

================================================================================
✅ ALL TESTS PASSED
================================================================================
```

## Step 7: Test Full Workflow

Submit a test workflow via the dashboard or CLI:

**Dashboard Method:**
1. Open `http://54.167.201.176:3000`
2. Go to "Talk to King AI" tab
3. Type: `"Implement a simple hello world API endpoint"`
4. Wait for approval prompt
5. Review PRD and approve
6. Check results

**CLI Method:**
```bash
# SSH to AWS
ssh -i king-ai-studio.pem ubuntu@54.167.201.176

# Submit workflow
kautilya workflow submit ralph-code-implementation \
  --input '{"task_description": "Create a /health endpoint that returns {\"status\": \"ok\"}"}'

# Monitor execution
kautilya workflow status <workflow_id>

# Stream logs
kautilya workflow logs <workflow_id> --follow
```

## Step 8: Verify Results

Check that Ralph executed successfully:

```bash
# View workflow results
kautilya workflow results <workflow_id>

# Check provenance in PostgreSQL
docker exec -it postgres psql -U postgres -d king_ai -c \
  "SELECT * FROM provenance WHERE workflow_id = '<workflow_id>';"

# View Ralph execution logs
ssh ubuntu@54.167.201.176 'ls -lh /tmp/ralph_*.log'
```

## Step 9: Enable in Dashboard

Update the dashboard to show Ralph Code Agent capability:

```bash
# Edit dashboard config
cd /home/ubuntu/king-ai-v3/agentic-framework-main/dashboard

# Add Ralph to capabilities list in src/config/capabilities.js
# (This step depends on your dashboard implementation)
```

## Step 10: Production Configuration

For production use, configure these additional settings:

### 1. Approval Workflow

Set up approval notifications:
```yaml
# In orchestrator config
approval_notifications:
  - type: email
    recipients: ["admin@example.com"]
  - type: slack
    webhook_url: "https://hooks.slack.com/..."
```

### 2. Rate Limiting

Limit Ralph executions per user:
```yaml
# In orchestrator config
rate_limits:
  ralph_code_agent:
    max_per_hour: 10
    max_per_day: 50
```

### 3. Resource Quotas

Set CPU/memory limits for Ralph executions:
```yaml
# In docker-compose.yml for orchestrator
services:
  orchestrator:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### 4. Monitoring

Set up monitoring for Ralph executions:
```bash
# Prometheus metrics
curl http://localhost:8000/metrics | grep ralph_code_agent

# Grafana dashboard
# Import dashboard from examples/monitoring/ralph-dashboard.json
```

## Rollback Plan

If issues occur, rollback the deployment:

```bash
# SSH to AWS
ssh -i king-ai-studio.pem ubuntu@54.167.201.176

# Remove skill
rm -rf /home/ubuntu/king-ai-v3/agentic-framework-main/code-exec/skills/ralph_code_agent

# Remove manifest
rm /home/ubuntu/king-ai-v3/agentic-framework-main/orchestrator/manifests/ralph-code-implementation.yaml

# Restart services
docker-compose restart orchestrator code-executor subagent-manager
```

## Post-Deployment Checklist

- [ ] Environment variables configured
- [ ] Skill files deployed to Code Executor
- [ ] Workflow manifest deployed to Orchestrator
- [ ] SSH connectivity tested
- [ ] Unit tests passing
- [ ] End-to-end workflow tested
- [ ] Dashboard updated
- [ ] Monitoring configured
- [ ] Documentation updated
- [ ] Team notified

## Security Considerations

1. **SSH Key Security**: Store SSH key in secure vault, mount read-only in containers
2. **Approval Gates**: Always require approval for production deployments
3. **Network Security**: Use VPC/security groups to restrict Ralph server access
4. **Audit Logging**: Enable full audit logging for all Ralph executions
5. **Code Review**: Review all Ralph-generated code before merging to main branch

## Next Steps

1. **Train your team** on Ralph Code Agent usage
2. **Create templates** for common code tasks (API endpoints, database models, etc.)
3. **Set up CI/CD** to automatically test Ralph-generated code
4. **Monitor usage** and iterate on PRD templates based on success rates
5. **Expand capabilities** by adding more Ralph skills (debugging, refactoring, etc.)

## Support

For issues or questions:
- Check logs: `docker logs orchestrator --tail 100`
- Review docs: `code-exec/skills/ralph_code_agent/README.md`
- Test locally: `python test_handler.py`
- Contact: [Your support channel]
