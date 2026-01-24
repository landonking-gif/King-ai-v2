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
- [ ] SSH key accessible
- [ ] Project files ready for upload

## Deployment Steps

### 1. Initial Server Access
- [ ] SSH connection established: `ssh -i "key.pem" ubuntu@EC2_IP`
- [ ] Server responding to commands

### 2. Upload Project Files
- [ ] Create deployment archive locally:
  ```bash
  cd king-ai-v3/agentic-framework-main
  tar -czf king-ai-v3.tar.gz --exclude="node_modules" --exclude=".git" --exclude="__pycache__" .
  ```
- [ ] Upload archive to server:
  ```bash
  scp -i "key.pem" king-ai-v3.tar.gz ubuntu@EC2_IP:~/
  ```

### 3. Upload Deployment Script
- [ ] Upload deployment script:
  ```bash
  scp -i "key.pem" scripts/deploy_aws.sh ubuntu@EC2_IP:~/
  ```

### 4. Run Deployment
- [ ] Make script executable: `chmod +x deploy_aws.sh`
- [ ] Execute deployment: `./deploy_aws.sh`
- [ ] Monitor deployment progress in terminal
- [ ] Check for any error messages

### 5. Post-Deployment Verification

#### Service Health Checks
- [ ] Orchestrator: `curl http://localhost:8000/api/health`
- [ ] MCP Gateway: `curl http://localhost:8080/health`
- [ ] Memory Service: `curl http://localhost:8002/health`
- [ ] Subagent Manager: `curl http://localhost:8001/health`
- [ ] Dashboard: `curl http://localhost:3000`

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
- [ ] Open browser to `http://EC2_IP:3000`
- [ ] Dashboard loads without errors
- [ ] "Talk to King AI" tab accessible
- [ ] Chat interface functional

### 6. Optional Enhancements

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
- [ ] Check Docker installation: `docker --version`
- [ ] Review deployment logs: `cat deploy_*.log`

#### Services Not Starting
- [ ] Check port conflicts: `netstat -tlnp | grep :8000`
- [ ] Verify environment variables: `cat */.env`
- [ ] Check service logs: `tail -f orchestrator.log`
- [ ] Validate Docker containers: `docker ps -a`

#### LLM Not Working
- [ ] Check Ollama container: `docker logs ollama`
- [ ] Verify model download: `docker exec ollama ollama list`
- [ ] Test Ollama API: `curl http://localhost:11434/api/generate -d '{"model":"llama3.1:70b","prompt":"test"}'`

#### Dashboard Not Loading
- [ ] Check Node.js installation: `node --version`
- [ ] Verify build process: `cd dashboard && npm run build`
- [ ] Check dashboard logs: `tail -f dashboard/dashboard.log`
- [ ] Test proxy configuration: `curl http://localhost:3000`

### Emergency Recovery

#### Restart All Services
```bash
# Stop all services
pkill -f "python.*service/main.py"
pkill -f "npm"
docker-compose down

# Restart infrastructure
docker-compose up -d postgres redis chroma

# Wait for databases
sleep 30

# Restart services
./deploy_aws.sh
```

#### Full System Reset
```bash
# Backup current state
cp -r king-ai-v3 king-ai-v3.backup

# Clean restart
docker-compose down -v
docker system prune -f
rm -rf king-ai-v3 venv

# Re-run deployment
./deploy_aws.sh
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
./monitor.sh

# Docker monitoring
docker stats
docker logs -f orchestrator
```

### Backup Strategy
```bash
# Database backup
docker exec postgres pg_dump -U agent_user agentic_framework > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz */.env

# Full system backup
tar -czf full_backup_$(date +%Y%m%d).tar.gz --exclude="node_modules" --exclude="venv" king-ai-v3/
```

## Support

If you encounter issues not covered here:
1. Check the logs: `tail -f *.log`
2. Review the USER_GUIDE.md for detailed instructions
3. Check GitHub issues for similar problems
4. Contact support@king-ai-studio.me

---

**Deployment completed successfully?** ✅ Check off all items and enjoy your King AI v3 system!