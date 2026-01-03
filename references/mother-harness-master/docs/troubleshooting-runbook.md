# Troubleshooting Runbook

## Overview

This runbook provides step-by-step troubleshooting procedures for common issues in the Mother-Harness platform.

**Last Updated:** December 22, 2024
**Audience:** On-call engineers, DevOps, Support

---

## Quick Reference

| Issue | First Check | Common Fix | Runbook Section |
|-------|-------------|------------|-----------------|
| Service down | Container status | Restart service | [Service Down](#service-down) |
| High errors | Logs | Check dependencies | [High Error Rate](#high-error-rate) |
| Slow performance | Metrics dashboard | Check resources | [Performance Issues](#performance-issues) |
| Redis issues | Connection test | Restart Redis | [Redis Problems](#redis-problems) |
| Auth failures | Token validity | Check Google OAuth | [Authentication Issues](#authentication-issues) |
| Task failures | Agent logs | Check Ollama | [Task Execution Failures](#task-execution-failures) |

---

## General Troubleshooting Steps

### Step 1: Verify System Health

```bash
# Check all services
docker-compose ps

# Check health endpoints
curl http://localhost:8000/health      # Orchestrator
curl http://localhost:8080/health      # Docling
curl http://localhost:3000             # Dashboard

# Check Redis
redis-cli PING
```

**Expected Output:**
- All services: `Up` status
- Health endpoints: `200 OK` with `{"status":"healthy"}`
- Redis: `PONG`

---

### Step 2: Check Logs

```bash
# All services (last 50 lines)
docker-compose logs --tail=50

# Specific service
docker-compose logs --tail=100 orchestrator
docker-compose logs --tail=100 docling
docker-compose logs --tail=100 redis-stack

# Follow logs in real-time
docker-compose logs -f orchestrator

# Search for errors
docker-compose logs orchestrator | grep -i error
docker-compose logs orchestrator | grep -i exception
```

---

### Step 3: Check Resources

```bash
# Container resource usage
docker stats

# Host resource usage
top
df -h
free -m

# Redis memory
redis-cli INFO memory
redis-cli INFO stats
```

---

## Service Down

### Symptoms
- Health check endpoint not responding
- Container not running
- Service unavailable errors

### Diagnosis

**1. Check container status:**
```bash
docker-compose ps orchestrator
```

**Possible States:**
- `Up` - Service running (check logs for errors)
- `Exit 0` - Clean shutdown (needs restart)
- `Exit 1` - Crashed (check logs for cause)
- `Restarting` - Crash loop (fix underlying issue)

**2. Check logs for crash reason:**
```bash
docker-compose logs --tail=100 orchestrator | grep -i error
```

**Common Errors:**
- `Cannot connect to Redis` → [Redis connection issue](#redis-connection-error)
- `Port already in use` → Another process using port
- `Out of memory` → Container memory limit too low
- `Module not found` → Build issue or missing dependency

---

### Resolution

**Quick Fix - Restart Service:**
```bash
docker-compose restart orchestrator
docker-compose logs -f orchestrator
```

**If restart doesn't work:**

**Option 1: Rebuild and restart:**
```bash
docker-compose build orchestrator
docker-compose up -d orchestrator
```

**Option 2: Full restart:**
```bash
docker-compose down
docker-compose up -d
```

**Option 3: Check for port conflicts:**
```bash
# Linux/Mac
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

### Verification

```bash
# Service is up
docker-compose ps orchestrator
# State should be "Up"

# Health check passes
curl http://localhost:8000/health
# Should return {"status":"healthy"}

# Can create task
curl -X POST http://localhost:8000/api/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}'
```

---

## High Error Rate

### Symptoms
- Elevated 5xx errors in metrics
- Many failed requests
- User reports of errors

### Diagnosis

**1. Identify error pattern:**
```bash
# Recent errors
docker-compose logs --tail=200 orchestrator | grep "500\|ERROR"

# Count errors by type
docker-compose logs orchestrator | grep ERROR | cut -d' ' -f4- | sort | uniq -c | sort -rn
```

**2. Check error rate:**
```bash
# Get metrics summary
curl http://localhost:8000/api/metrics/summary \
  -H "Authorization: Bearer $TOKEN"
```

**3. Identify affected endpoints:**
Check logs for patterns:
- All endpoints → Infrastructure issue
- Specific endpoint → Code bug or dependency issue
- Random endpoints → Race condition or resource exhaustion

---

### Common Causes and Fixes

#### Redis Connection Error

**Error:** `Error: Redis connection failed`

**Check:**
```bash
docker-compose ps redis-stack
redis-cli PING
```

**Fix:**
```bash
# Restart Redis
docker-compose restart redis-stack

# Restart dependent services
docker-compose restart orchestrator
```

---

#### Ollama Unavailable

**Error:** `Failed to connect to Ollama` or `Model inference failed`

**Check:**
```bash
# Check Ollama health (if local)
curl http://core2:11434/api/tags

# Or check environment variable
docker-compose exec orchestrator env | grep OLLAMA
```

**Fix:**
```bash
# If Ollama is local, restart it
ssh core2 "sudo systemctl restart ollama"

# If using cloud fallback, verify API key
docker-compose exec orchestrator env | grep OLLAMA_CLOUD_API_KEY

# Restart orchestrator to reconnect
docker-compose restart orchestrator
```

---

#### n8n Workflow Failure

**Error:** `n8n webhook failed` or `Agent execution timed out`

**Check:**
```bash
curl http://n8n:5678/healthz
```

**Fix (Temporary - Use Local Agents):**
```bash
# n8n is optional, local agents work without it
# Errors will auto-fallback to local execution
# Monitor for successful local execution in logs
docker-compose logs -f orchestrator | grep "agent execution"
```

---

#### JWT Validation Error

**Error:** `Invalid token` or `Token expired`

**Check:**
```bash
# Verify JWT_SECRET is set
docker-compose exec orchestrator env | grep JWT_SECRET

# Check token expiration
# Decode JWT at https://jwt.io
```

**Fix:**
```bash
# If JWT_SECRET changed, users need new tokens
# Verify secret hasn't changed
cat .env | grep JWT_SECRET

# If secret is correct, token may be expired
# User needs to re-authenticate
```

---

## Performance Issues

### Symptoms
- Slow response times (P95 >1s)
- Timeouts
- High latency alerts

### Diagnosis

**1. Check response times:**
```bash
# Test health endpoint speed
time curl http://localhost:8000/health

# Should complete in <100ms
```

**2. Check system resources:**
```bash
docker stats

# Look for:
# - CPU >80%
# - Memory near limit
# - High disk I/O
```

**3. Check Redis performance:**
```bash
redis-cli INFO stats | grep instantaneous_ops_per_sec
redis-cli --latency
redis-cli SLOWLOG GET 10
```

---

### Resolution

#### High CPU Usage

**Cause:** Too many concurrent tasks or inefficient code

**Fix:**
```bash
# Check concurrent tasks
curl http://localhost:8000/api/tasks?status=running \
  -H "Authorization: Bearer $TOKEN"

# If many tasks running, they'll complete
# If stuck, check for infinite loops in logs

# Temporarily limit concurrency (restart required)
# Edit docker-compose.yml to reduce max workers
```

---

#### High Memory Usage

**Cause:** Memory leak or too many tasks in memory

**Check:**
```bash
docker stats orchestrator

# If memory climbing steadily = leak
# If memory high but stable = normal load
```

**Fix (Immediate):**
```bash
# Restart service to free memory
docker-compose restart orchestrator
```

**Fix (Long-term):**
- Monitor memory growth rate
- Profile application for leaks
- Increase container memory limit if needed

---

#### Slow Redis Queries

**Cause:** Large keys, unoptimized queries, or memory pressure

**Check:**
```bash
redis-cli SLOWLOG GET 10
redis-cli INFO memory
redis-cli --bigkeys
```

**Fix:**
```bash
# If memory full, enable eviction
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# If specific slow queries, optimize them
# Check for KEYS command (use SCAN instead)

# Clear cache if safe to do so
redis-cli FLUSHDB
```

---

## Redis Problems

### Redis Connection Error

**Symptoms:**
- All services reporting Redis errors
- `ECONNREFUSED` in logs

**Diagnosis:**
```bash
docker-compose ps redis-stack
redis-cli PING
```

**Resolution:**
```bash
# Restart Redis
docker-compose restart redis-stack

# Check logs for errors
docker-compose logs redis-stack

# Restart dependent services
docker-compose restart orchestrator docling
```

---

### Redis Out of Memory

**Symptoms:**
- `OOM command not allowed` errors
- High memory usage alerts

**Diagnosis:**
```bash
redis-cli INFO memory | grep used_memory_human
redis-cli INFO memory | grep maxmemory
```

**Resolution:**

**Option 1: Clear non-critical data:**
```bash
# Check largest keys
redis-cli --bigkeys

# Delete old metrics (if safe)
redis-cli DEL "metrics:activity:2024-01-*"

# Flush cache (if using Redis for caching)
redis-cli FLUSHDB
```

**Option 2: Increase memory limit:**
```bash
# Edit docker-compose.yml
# redis-stack:
#   command: redis-stack-server --maxmemory 8gb

docker-compose up -d redis-stack
```

**Option 3: Enable eviction:**
```bash
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

### Redis Slow Performance

**Symptoms:**
- Slow query alerts
- High latency from Redis

**Diagnosis:**
```bash
redis-cli --latency
redis-cli SLOWLOG GET 10
redis-cli INFO stats
```

**Resolution:**
```bash
# Check for long-running commands
redis-cli CLIENT LIST

# Kill slow clients if needed
redis-cli CLIENT KILL TYPE normal

# Optimize indexes
# Rebuild search indexes if using RediSearch
redis-cli FT.DROPINDEX task_idx
# Restart orchestrator to recreate
```

---

## Authentication Issues

### Users Cannot Login

**Symptoms:**
- Google OAuth redirect fails
- "Authentication failed" errors

**Diagnosis:**

**1. Check Google OAuth configuration:**
```bash
docker-compose exec orchestrator env | grep GOOGLE_CLIENT_ID
```

**2. Check allowed domains:**
```bash
docker-compose exec orchestrator env | grep GOOGLE_ALLOWED_DOMAINS
```

**3. Check orchestrator logs:**
```bash
docker-compose logs orchestrator | grep -i "auth\|oauth"
```

**Common Issues:**
- `GOOGLE_CLIENT_ID` not set or invalid
- User domain not in `GOOGLE_ALLOWED_DOMAINS`
- Google OAuth consent screen not configured

---

### Resolution

**Fix 1: Verify environment variables:**
```bash
cat .env | grep GOOGLE_

# Should have:
# GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
# GOOGLE_ALLOWED_DOMAINS=example.com,example.org
```

**Fix 2: Add user's domain to allowed list:**
```bash
# Edit .env
GOOGLE_ALLOWED_DOMAINS=example.com,newdomain.com

# Restart
docker-compose restart orchestrator
```

**Fix 3: Check Google Cloud Console:**
- Verify OAuth consent screen is published
- Verify authorized redirect URIs include your domain
- Check API quotas haven't been exceeded

---

### JWT Token Issues

**Symptoms:**
- "Invalid token" errors
- Users get logged out frequently

**Diagnosis:**
```bash
# Check JWT_SECRET is set
docker-compose exec orchestrator env | grep JWT_SECRET

# Decode a user's token at https://jwt.io
# Check expiration time (exp claim)
```

**Resolution:**

**If JWT_SECRET changed:**
```bash
# All users need to re-authenticate
# No fix except re-login
```

**If tokens expiring too quickly:**
```bash
# Check token expiration time in code
# services/orchestrator/src/server.ts
# Look for jwt.sign() calls with expiresIn option

# Increase expiration if needed (requires code change)
```

---

## Task Execution Failures

### All Tasks Failing

**Symptoms:**
- >20% task failure rate
- Consistent failures across all task types

**Diagnosis:**
```bash
# Check recent task failures
curl http://localhost:8000/api/tasks?status=failed \
  -H "Authorization: Bearer $TOKEN"

# Check orchestrator logs
docker-compose logs --tail=100 orchestrator | grep -i "task\|execution\|error"
```

**Common Causes:**
- Ollama down
- n8n down (if not using local fallback)
- Agent code errors
- Redis connection issues

**Resolution:**

**1. Check Ollama:**
```bash
curl http://core2:11434/api/tags

# If down, restart
ssh core2 "sudo systemctl restart ollama"
```

**2. Verify local agent fallback:**
```bash
docker-compose logs orchestrator | grep "fallback"

# Should see "Falling back to local agent execution"
```

**3. Check Redis connection:**
```bash
redis-cli PING
docker-compose restart redis-stack
```

---

### Specific Agent Type Failing

**Symptoms:**
- Only "coder" agent failing, others work
- Consistent error for specific agent

**Diagnosis:**
```bash
# Check agent-specific logs
docker-compose logs orchestrator | grep "agent\":\"coder"

# Check if agent requires specific tool
# Review agent implementation
```

**Resolution:**
```bash
# Check if agent has required dependencies
# E.g., coder agent might need git, npm, etc.

# Verify in orchestrator container
docker-compose exec orchestrator which git
docker-compose exec orchestrator which npm

# If missing, add to Dockerfile and rebuild
```

---

### Task Stuck in "Running" State

**Symptoms:**
- Task never completes
- Status stuck on same step
- No progress for >30 minutes

**Diagnosis:**
```bash
# Get task details
curl http://localhost:8000/api/tasks/{task_id} \
  -H "Authorization: Bearer $TOKEN"

# Check logs for task ID
docker-compose logs orchestrator | grep "{task_id}"
```

**Resolution:**

**Option 1: Wait (if recent):**
- Complex tasks can take 10-20 minutes
- Check if there's progress in logs

**Option 2: Manually fail task:**
```bash
# Currently no API endpoint for this
# Would need to update Redis directly
redis-cli HSET "task:{task_id}" status failed

# Or restart orchestrator (will mark as failed on restart)
docker-compose restart orchestrator
```

---

## Document Processing Issues

### Docling Service Down

**Symptoms:**
- Document upload/processing fails
- Health check failing on port 8080

**Diagnosis:**
```bash
docker-compose ps docling
curl http://localhost:8080/health
docker-compose logs --tail=50 docling
```

**Resolution:**
```bash
# Restart docling service
docker-compose restart docling

# If issues persist, check file system
df -h /core4/libraries

# Check permissions
ls -la /core4/libraries
```

---

### PDF Processing Failures

**Symptoms:**
- PDF documents fail to process
- "Failed to extract text" errors

**Diagnosis:**
```bash
docker-compose logs docling | grep -i "pdf\|error"

# Check if PDF fallback is working
docker-compose logs docling | grep "fallback"
```

**Resolution:**

**Docling has built-in PDF fallback:**
- If Docling API fails, uses pdf-parse library
- Check logs for fallback execution
- Most PDFs should process with fallback

**If still failing:**
```bash
# Check if file is accessible
docker-compose exec docling ls -la /core4/libraries/{library_id}/

# Check file permissions
# Files should be readable by docling container

# Try reprocessing
# API endpoint: POST /api/libraries/{id}/rescan
```

---

## Backup and Restore Issues

### Backup Failed

**Symptoms:**
- Backup script exited with error
- No backup file created

**Diagnosis:**
```bash
# Run backup manually to see error
./scripts/backup-redis.sh

# Check disk space
df -h /var/backups/mother-harness/redis/

# Check Redis connection
redis-cli PING

# Check permissions
ls -la /var/backups/mother-harness/redis/
```

**Common Errors:**
- `No space left on device` → Clean up old backups or expand disk
- `Permission denied` → Fix directory permissions
- `Redis connection refused` → Redis is down

**Resolution:**
```bash
# Fix permissions
sudo chown -R $(whoami) /var/backups/mother-harness/redis/

# Clean old backups
ls -lt /var/backups/mother-harness/redis/ | tail -n +31 | awk '{print $9}' | xargs rm -f

# Retry backup
./scripts/backup-redis.sh
```

---

### Restore Failed

**Symptoms:**
- Restore script error
- Data not restored

**Diagnosis:**
```bash
# Check backup file integrity
tar -tzf /var/backups/mother-harness/redis/redis_backup_YYYYMMDD_HHMMSS.tar.gz

# Check if backup contains dump.rdb
tar -xzf redis_backup_YYYYMMDD_HHMMSS.tar.gz -O | tar -t

# Verify Redis is stopped (for restore)
docker-compose ps redis-stack
```

**Resolution:**
```bash
# Ensure Redis is stopped before restore
docker-compose stop redis-stack

# Run restore
./scripts/restore-redis.sh redis_backup_YYYYMMDD_HHMMSS.tar.gz

# Start Redis
docker-compose start redis-stack

# Verify data
redis-cli DBSIZE
```

---

## Network and Connectivity Issues

### Inter-Service Communication Failures

**Symptoms:**
- "Cannot connect to http://docling:8080" errors
- Services can't reach each other

**Diagnosis:**
```bash
# Check Docker network
docker network ls
docker network inspect mother-harness_default

# Test connectivity from within container
docker-compose exec orchestrator ping docling
docker-compose exec orchestrator curl http://docling:8080/health
```

**Resolution:**
```bash
# Restart Docker network
docker-compose down
docker-compose up -d

# Verify all services on same network
docker network inspect mother-harness_default | grep Name
```

---

### External Service Unreachable

**Symptoms:**
- Cannot reach Ollama on core2
- Cannot reach n8n

**Diagnosis:**
```bash
# From host
ping core2
curl http://core2:11434/api/tags

# From container
docker-compose exec orchestrator ping core2
docker-compose exec orchestrator curl http://core2:11434/api/tags
```

**Resolution:**
```bash
# Check host network configuration
# Verify DNS resolution
nslookup core2

# Check firewall rules
sudo iptables -L

# Verify service is running on remote host
ssh core2 "systemctl status ollama"
```

---

## Dashboard Issues

### Dashboard Not Loading

**Symptoms:**
- Blank page or loading spinner
- Console errors in browser

**Diagnosis:**
```bash
# Check dashboard service
docker-compose ps dashboard
docker-compose logs dashboard

# Check browser console for errors
# (Open DevTools → Console)
```

**Resolution:**
```bash
# Restart dashboard
docker-compose restart dashboard

# Clear browser cache
# Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)

# Rebuild if needed
docker-compose build dashboard
docker-compose up -d dashboard
```

---

### Dashboard Shows No Data

**Symptoms:**
- Dashboard loads but shows empty state
- "No tasks found" or similar

**Diagnosis:**
```bash
# Check if orchestrator is reachable from browser
curl http://localhost:8000/health

# Check API connectivity from dashboard container
docker-compose exec dashboard curl http://orchestrator:8000/health

# Check browser network tab for API calls
# (DevTools → Network → XHR)
```

**Resolution:**
```bash
# Verify ORCHESTRATOR_URL in dashboard env
docker-compose exec dashboard env | grep ORCHESTRATOR_URL

# Should point to orchestrator service
# ORCHESTRATOR_URL=http://orchestrator:8000

# Restart dashboard
docker-compose restart dashboard
```

---

## Deployment Issues

### Build Failures

**Symptoms:**
- `docker-compose build` fails
- TypeScript compilation errors

**Common Errors:**

**Missing dependencies:**
```bash
# Error: Cannot find module '@mother-harness/shared'
# Fix: Build shared package first
cd services/shared && npm run build
cd ../.. && docker-compose build
```

**TypeScript errors:**
```bash
# Fix type errors in code
# Or temporarily skip type checking (not recommended for production)
# Edit tsconfig.json: "skipLibCheck": true
```

---

### Container Won't Start After Deploy

**Symptoms:**
- Container exits immediately after `docker-compose up`
- Exit code 1

**Diagnosis:**
```bash
docker-compose logs --tail=50 orchestrator

# Look for:
# - "Module not found" → Build issue
# - "Cannot find .env" → Missing env file
# - "Port already in use" → Port conflict
# - Segmentation fault → Memory issue
```

**Resolution:**
```bash
# Rebuild from scratch
docker-compose build --no-cache orchestrator
docker-compose up -d orchestrator

# Check environment file
cat .env | head -5

# Free up port if needed
docker-compose down
docker-compose up -d
```

---

## Advanced Diagnostics

### Enable Debug Logging

```bash
# Edit docker-compose.yml
services:
  orchestrator:
    environment:
      - LOG_LEVEL=debug  # Change from 'info' to 'debug'

# Restart
docker-compose restart orchestrator

# View detailed logs
docker-compose logs -f orchestrator
```

---

### Memory Profiling

```bash
# Get heap snapshot (requires node inspect)
docker-compose exec orchestrator kill -SIGUSR2 $(pgrep node)

# Check for memory leaks
docker stats orchestrator
# Monitor over 30 minutes, should stabilize not climb continuously
```

---

### Redis Debugging

```bash
# Monitor all Redis commands
redis-cli MONITOR

# Get Redis info
redis-cli INFO all

# Check connected clients
redis-cli CLIENT LIST

# Analyze memory usage
redis-cli --bigkeys

# Check slow queries
redis-cli SLOWLOG GET 25
```

---

## When to Escalate

Escalate to engineering team if:
- Issue persists after following runbook
- Data corruption suspected
- Security incident detected
- Unknown error pattern
- Multiple services affected simultaneously
- Need code changes to resolve

See: [On-Call Procedures - Escalation](oncall-procedures.md#escalation-paths)

---

## Emergency Rollback

If deployment causes critical issues:

```bash
# Stop current version
docker-compose down

# Checkout previous version
git log --oneline -5  # Find previous commit
git checkout <previous-commit-hash>

# Rebuild and deploy
docker-compose build
docker-compose up -d

# Verify services
curl http://localhost:8000/health

# Restore from backup if data issue
./scripts/restore-redis.sh <backup-file>
```

---

## Maintenance Tasks

### Clean Up Docker Resources

```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes (CAREFUL!)
docker volume prune -f

# Remove unused networks
docker network prune -f
```

---

### Rotate Logs

```bash
# Docker logs can grow large
# Configure log rotation in docker-compose.yml:
services:
  orchestrator:
    logging:
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Useful Commands Reference

```bash
# Quick Health Check
./scripts/health-check.sh  # Create this script

# View all service logs
docker-compose logs --tail=50

# Restart all services
docker-compose restart

# Rebuild specific service
docker-compose build orchestrator && docker-compose up -d orchestrator

# Check Redis memory
redis-cli INFO memory | grep used_memory_human

# Test authentication
curl -X POST http://localhost:8000/api/auth/login

# Get metrics summary
curl http://localhost:8000/api/metrics/summary -H "Authorization: Bearer $TOKEN"

# List running tasks
curl http://localhost:8000/api/tasks?status=running -H "Authorization: Bearer $TOKEN"

# Backup Redis
./scripts/backup-redis.sh

# Restore Redis
./scripts/restore-redis.sh <backup-file>
```

---

## Related Documentation

- [On-Call Procedures](oncall-procedures.md)
- [Alerting Thresholds](alerting-thresholds.md)
- [Deployment Checklist](../DEPLOYMENT_CHECKLIST.md)
- [Backup and Restore](backup-restore.md)
- [Security Baseline](security-baseline.md)
- [API Reference](api-reference.md)

---

*Last Updated: December 22, 2024*
*Troubleshooting Runbook Version: 1.0*
*Next Review: February 1, 2025*
