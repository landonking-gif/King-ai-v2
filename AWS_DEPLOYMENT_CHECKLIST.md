# King AI v3 - AWS Deployment Checklist

## Pre-Deployment Checklist

### AWS Infrastructure
- [ ] EC2 instance launched (t3.medium or larger recommended)
- [ ] Ubuntu 22.04 LTS AMI selected
- [ ] Security group configured with ports:
  - [ ] 22 (SSH)
  - [ ] 80 (HTTP)
  - [ ] 443 (HTTPS - optional)
  - [ ] 3000 (Dashboard)
  - [ ] 8000 (Orchestrator API)
  - [ ] 8001 (Subagent Manager)
  - [ ] 8002 (Memory Service)
  - [ ] 8080 (MCP Gateway)
  - [ ] 11434 (Ollama)
- [ ] SSH key pair downloaded (.pem file)
- [ ] Public IP/DNS noted

### Local Environment
- [ ] Git repository cloned
- [ ] SSH key accessible at default location (`$HOME\.ssh\king-ai-studio.pem`)
- [ ] PowerShell available (Windows) or compatible shell
- [ ] Project files ready for deployment

## Deployment Steps

### 1. Single Command Deployment
- [ ] Run the complete deployment script:
  ```powershell
  .\deploy.ps1 -IP "YOUR_EC2_PUBLIC_IP"
  ```
- [ ] Monitor deployment progress (8 steps total)
- [ ] Wait for completion (typically 10-15 minutes)

### Alternative Usage
```powershell
# With custom SSH key path
.\deploy.ps1 -IP "YOUR_EC2_PUBLIC_IP" -KeyPath "C:\path\to\your\key.pem"

# Skip health checks (for faster deployment)
.\deploy.ps1 -IP "YOUR_EC2_PUBLIC_IP" -SkipHealthChecks

# Verbose output
.\deploy.ps1 -IP "YOUR_EC2_PUBLIC_IP" -Verbose
```

### 2. Post-Deployment Verification

#### Automated Health Checks
The deployment script automatically performs comprehensive health checks including:
- [ ] Individual service health (5 services)
- [ ] Nginx proxy endpoints (4 endpoints)
- [ ] Overall deployment success status

#### Manual Health Checks (if needed)
- [ ] Main endpoint: `curl http://YOUR_IP/health`
- [ ] API docs: `curl http://YOUR_IP/docs`
- [ ] Dashboard: `curl http://YOUR_IP/` (may show fallback message if not built)

#### LLM Verification
- [ ] Ollama service: `curl http://localhost:11434/api/tags`
- [ ] Model loaded: Check for "llama3.1:70b" in response
- [ ] Chat API test:
  ```bash
  curl -X POST http://localhost:8000/api/chat/message \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello King AI", "session_id": "test"}'
  ```

#### Dashboard Access
- [ ] Open browser to `http://EC2_IP/` (via Nginx proxy)
- [ ] Dashboard loads without errors (or shows friendly message if not built)
- [ ] API docs accessible at `http://EC2_IP/docs`

### 3. Optional Enhancements

#### Domain Configuration
- [ ] Domain purchased/available
- [ ] DNS A record pointing to EC2 IP
- [ ] Nginx configured for domain (if not using load balancer)

#### SSL Certificate
- [ ] AWS Load Balancer configured (recommended)
- [ ] Or Let's Encrypt certificate installed
- [ ] HTTPS redirect configured

#### Monitoring Setup
- [ ] Monitoring script tested: `./monitor.sh`
- [ ] Log rotation configured
- [ ] Backup strategy implemented

## Troubleshooting

### Common Issues

#### Deployment Script Fails
- [ ] Check available disk space: `df -h`
- [ ] Verify internet connectivity: `ping google.com`
- [ ] Check SSH key permissions: `ls -la ~/.ssh/king-ai-studio.pem`
- [ ] Review deployment logs on server: `tail -f /tmp/*.log`

#### Services Not Starting
- [ ] Check port conflicts: `netstat -tlnp | grep :8000`
- [ ] Verify environment variables: `cat */.env`
- [ ] Check service logs: `tail -f /tmp/orchestrator.log`
- [ ] Validate Python virtual environment: `source .venv/bin/activate && python --version`

#### LLM Not Working
- [ ] Check Ollama container: `docker logs ollama`
- [ ] Verify model download: `docker exec ollama ollama list`
- [ ] Test Ollama API: `curl http://localhost:11434/api/generate -d '{"model":"llama3.1:70b","prompt":"test"}'`

#### Dashboard Not Loading
- [ ] Check Node.js installation: `node --version`
- [ ] Verify build process: `cd dashboard && npm run build`
- [ ] Check dashboard logs: `tail -f /tmp/dashboard.log`
- [ ] Test proxy configuration: `curl http://localhost/health`

### Emergency Recovery

#### Restart All Services
```bash
# Stop all services
pkill -f "python.*service/main.py"
pkill -f "serve.*3000"
pkill -f "node.*3000"

# Restart infrastructure
sudo systemctl restart postgresql redis nginx

# Wait for services
sleep 30

# Re-run deployment
cd ~/agentic-framework-main
source .venv/bin/activate
# Restart individual services manually or re-run deploy.ps1
```

#### Full System Reset
```bash
# Backup current state (if needed)
cp -r agentic-framework-main agentic-framework-main.backup

# Clean restart
sudo systemctl stop postgresql redis nginx
sudo systemctl disable postgresql redis nginx
sudo apt-get remove -y postgresql redis-server nginx nodejs

# Remove all files
rm -rf agentic-framework-main dashboard deploy.zip server-deploy.sh .venv

# Re-run deployment from local machine
# .\deploy.ps1 -IP "YOUR_EC2_IP"
```

## Performance Optimization

### Instance Sizing
- **Minimum**: t3.medium (2 vCPU, 4GB RAM)
- **Recommended**: t3.large (2 vCPU, 8GB RAM)
- **Production**: t3.xlarge (4 vCPU, 16GB RAM)

### Monitoring Commands
```bash
# System resources
htop
df -h
free -h

# Service monitoring
ps aux | grep -E "(python|node|serve)"
tail -f /tmp/*.log

# Nginx monitoring
sudo tail -f /var/log/nginx/error.log
curl -s http://localhost/health | jq
```

### Backup Strategy
```bash
# Database backup
sudo -u postgres pg_dump -U agentic_user agentic_framework > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz */.env

# Full system backup
tar -czf full_backup_$(date +%Y%m%d).tar.gz --exclude="node_modules" --exclude="__pycache__" agentic-framework-main/
```

## Support

If you encounter issues not covered here:
1. Check the logs: `tail -f /tmp/*.log`
2. Review the USER_GUIDE.md for detailed instructions
3. Check GitHub issues for similar problems
4. Contact support@king-ai-studio.me

---

**Deployment completed successfully?** ✅ Check off all items and enjoy your King AI v3 system!