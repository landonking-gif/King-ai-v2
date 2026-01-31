# King AI v3 Deployment

## Single Deployment Script

**Use `deploy.ps1` for all deployments** - this is the comprehensive, production-ready deployment script that handles everything.

### Quick Start

```powershell
# Deploy to AWS EC2 instance
.\deploy.ps1 -IpAddress 52.90.206.76

# Skip package creation if deploy.zip already exists
.\deploy.ps1 -IpAddress 52.90.206.76 -SkipBuild
```

### What deploy.ps1 Does

1. **Package Creation**: Multi-threaded ZIP creation with exclusions
2. **File Upload**: Secure SCP upload with compression
3. **Server Setup**: Complete Ubuntu environment setup
4. **Service Deployment**: All services (Orchestrator, Memory, MCP, Dashboard, MoltBot)
5. **AI Setup**: Ollama + DeepSeek R1 7B model installation
6. **Health Checks**: Comprehensive service verification
7. **Logging**: Detailed deployment logs with timestamps

### Services Deployed

- **Orchestrator** (port 8000) - Main API service
- **Memory Service** (port 8002) - Data persistence
- **MCP Gateway** (port 8080) - Model Context Protocol
- **Subagent Manager** (port 8001) - AI subagents
- **Dashboard** (port 3000) - Web interface
- **MoltBot** (port 18789) - Multi-channel gateway
- **Ollama** (port 11434) - AI model runtime
- **Nginx** (port 80) - Reverse proxy

### Access Information

After successful deployment:
- **API**: http://[IP]/api
- **Docs**: http://[IP]/docs
- **Dashboard**: http://[IP]:3000
- **MoltBot UI**: http://[IP]:18789

### Troubleshooting

Check deployment logs:
```bash
ssh ubuntu@[IP] 'cat /tmp/deployment.log'
```

Service logs:
```bash
ssh ubuntu@[IP] 'tail -f /tmp/orchestrator.log'
ssh ubuntu@[IP] 'tail -f /tmp/moltbot.log'
```

### Previous Scripts Removed

The following fragmented scripts have been removed in favor of the single `deploy.ps1`:

- `quick-deploy.ps1` - Replaced by comprehensive deploy.ps1
- `server-setup.sh` - Functionality merged into enhanced-deploy.sh
- `simple-deploy.sh` - Service startup handled by enhanced-deploy.sh
- Various test scripts and old deployment artifacts

**Only use `deploy.ps1` for all deployments going forward.**