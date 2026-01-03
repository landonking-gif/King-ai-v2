# Deployment Readiness Checklist

## ✅ Completed Fixes

### 1. Dependencies
- ✅ Added missing `pino-pretty` to orchestrator devDependencies
- ✅ Removed unused `socket.io-client` from dashboard
- ✅ All workspace dependencies properly configured

### 2. Docker Configuration
- ✅ Fixed docker-compose.yml build contexts to use root directory
- ✅ All Dockerfiles properly reference workspace files
- ✅ pnpm-lock.yaml is optional (using `*` wildcard)

### 3. Environment Configuration
- ✅ Created `env.example` with all required variables
- ✅ Updated README with correct filename reference

### 4. WebSocket Support
- ✅ Added WebSocket endpoint handler in orchestrator
- ✅ Removed socket.io-client dependency mismatch

### 5. Code Fixes
- ✅ Added React import to dashboard layout
- ✅ All exports verified from shared package
- ✅ Fixed TypeScript CSS module type error in ApprovalMessage component
- ✅ Fixed phone number PII redaction regex (all 27 tests passing)
- ✅ Reordered PII patterns to prevent false matches

## 📋 Pre-Deployment Checklist

### Required Environment Variables
Before deploying, ensure `.env` file contains:
- `REDIS_PASSWORD` - Redis authentication password
- `REDIS_USERNAME` - Redis ACL user (for example: orchestrator)
- `JWT_SECRET` - Secret for signing auth tokens (must be rotated)
- `GOOGLE_CLIENT_ID` - Google OAuth client ID for sign-in
- `GOOGLE_ALLOWED_DOMAINS` - Comma-separated allowed domains (optional)
- `GOOGLE_ADMIN_EMAILS` - Comma-separated admin emails (optional)
- `GOOGLE_APPROVER_EMAILS` - Comma-separated approver emails (optional)
- `OLLAMA_LOCAL_URL` - Local Ollama instance URL (default: http://core2:11434)
- `OLLAMA_CLOUD_API_KEY` - Optional cloud API key
- `N8N_URL` - n8n instance URL (default: http://n8n:5678)
- `N8N_API_KEY` - Optional n8n API key
- `LIBRARIES_PATH` - Path to document libraries (default: /core4/libraries)

### Service Dependencies
1. **Redis Stack** - Must be running and healthy before other services
2. **Ollama** - Required for local model inference (can be on separate host)
3. **n8n** - Optional, for workflow-based agent execution
4. **File Storage** - Libraries path must be accessible to docling service

### Build Process
1. All services build from root directory context
2. Shared package builds first, then dependent services
3. Production dependencies only in final stage

### Health Checks
- Orchestrator: `GET http://localhost:8000/health`
- Redis: Health check configured in docker-compose
- Dashboard: Available on port 3000

### Secret Verification
- Confirm all secret placeholders (e.g., `CHANGE_ME`) are replaced in `.env`.
- Verify secrets are unique per environment and rotated before deployment.

## 🚀 Deployment Steps

1. **Pull latest changes from git** (REQUIRED before building)
   ```bash
   cd /path/to/mother-harness
   git pull origin master
   ```

2. **Copy environment file** (if not already configured)
   ```bash
   cp env.example .env
   # Edit .env with your production values
   ```

3. **Ensure Redis Stack is accessible**
   - Default: `redis://:password@redis-stack:6379`
   - Update REDIS_URL if using external Redis

4. **Configure Redis ACL users**
   - Rotate the default ACL passwords in `.env`
   - Apply ACL config via `redis-cli` (example):
     ```bash
     ACL SETUSER orchestrator on >${REDIS_ACL_ORCHESTRATOR_PASSWORD} +get +set +del +keys +json.* +ft.* +xadd +xread +xreadgroup +xack +hget +hset +hgetall +hincrby +hincrbyfloat +expire +exists +ping ~task:* ~project:* ~approval:* ~model_decision:* ~cost:* ~budget:* ~retry:*
     ```

5. **Build and start services**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

6. **Verify health**
   ```bash
   curl http://localhost:8002/health  # Orchestrator (note: port 8002)
   curl http://localhost:3000         # Dashboard
   ```

7. **Check logs**
   ```bash
   docker-compose logs -f orchestrator
   docker-compose logs -f dashboard
   ```

## ⚠️ Known Limitations

No known production blockers at this time. Optional services (n8n workflows and Docling API) can be added to extend automation and PDF/OCR fidelity as needed.

## 🔧 Post-Deployment Configuration

1. **Set up Redis ACL users**:
   ```bash
   ./scripts/setup-redis-acl.sh
   ```
2. **Verify health checks**:
   ```bash
   curl http://localhost:8000/health  # Orchestrator
   curl http://localhost:8080/health  # Docling
   curl http://localhost:3000         # Dashboard
   ```
3. **Create Libraries**: Use API or Redis directly to create library entries
4. **Configure n8n Workflows**: Import workflows from `n8n-workflows/` directory
5. **Set up Ollama Models**: Ensure required models are pulled on Ollama host
6. **Configure Google Authentication**: Set GOOGLE_CLIENT_ID and related env vars

## 📝 Notes

- All services use pnpm workspaces
- Build artifacts are in `dist/` directories
- Redis indexes are created automatically on first run
- Services gracefully handle missing optional dependencies (n8n, Ollama cloud)
