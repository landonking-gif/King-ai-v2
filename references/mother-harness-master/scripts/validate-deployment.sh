#!/bin/bash
#
# Deployment Validation Script
# Validates that Mother-Harness deployment meets minimum requirements
#
# Usage: ./scripts/validate-deployment.sh [environment]
# Example: ./scripts/validate-deployment.sh staging

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Mother-Harness Deployment Validation${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Helper functions
pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

section() {
    echo ""
    echo -e "${BLUE}## $1${NC}"
}

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    pass "Environment file loaded (.env)"
else
    fail "Environment file not found (.env)"
    echo "  → Run: cp env.example .env"
fi

# ===========================================
# Section 1: Environment Variables
# ===========================================
section "Environment Variables"

# Check required secrets
REQUIRED_SECRETS=(
    "JWT_SECRET"
    "REDIS_PASSWORD"
    "REDIS_ACL_ORCHESTRATOR_PASSWORD"
    "REDIS_ACL_DOCLING_PASSWORD"
    "REDIS_ACL_AGENTS_PASSWORD"
    "REDIS_ACL_DASHBOARD_PASSWORD"
)

INVALID_VALUES=("" "CHANGE_ME" "changeme" "password" "default")

for SECRET in "${REQUIRED_SECRETS[@]}"; do
    VALUE="${!SECRET}"
    if [ -z "$VALUE" ]; then
        fail "$SECRET is not set"
    elif [[ " ${INVALID_VALUES[@]} " =~ " ${VALUE} " ]]; then
        fail "$SECRET has invalid/default value"
    elif [ ${#VALUE} -lt 16 ]; then
        warn "$SECRET is less than 16 characters (weak)"
    else
        pass "$SECRET is set and appears valid"
    fi
done

# Check optional but recommended variables
OPTIONAL_VARS=(
    "GOOGLE_CLIENT_ID"
    "OLLAMA_LOCAL_URL"
    "LIBRARIES_PATH"
)

for VAR in "${OPTIONAL_VARS[@]}"; do
    VALUE="${!VAR}"
    if [ -z "$VALUE" ]; then
        warn "$VAR is not set (optional but recommended)"
    else
        pass "$VAR is set"
    fi
done

# ===========================================
# Section 2: Service Health Checks
# ===========================================
section "Service Health Checks"

# Determine base URLs
ORCHESTRATOR_URL=${ORCHESTRATOR_URL:-http://localhost:8000}
DOCLING_URL=${DOCLING_URL:-http://localhost:8080}
DASHBOARD_URL=${DASHBOARD_URL:-http://localhost:3000}
REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

# Check if services are running via Docker
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
    if docker-compose ps 2>/dev/null | grep -q "Up"; then
        pass "Docker Compose services are running"
    else
        warn "Docker Compose services may not be running"
        echo "  → Run: docker-compose up -d"
    fi
else
    warn "Docker/Docker Compose not found (skipping container checks)"
fi

# Health check: Orchestrator
if curl -sf "$ORCHESTRATOR_URL/health" > /dev/null 2>&1; then
    HEALTH_STATUS=$(curl -s "$ORCHESTRATOR_URL/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        pass "Orchestrator is healthy ($ORCHESTRATOR_URL)"
    else
        warn "Orchestrator returned status: $HEALTH_STATUS"
    fi
else
    fail "Orchestrator health check failed ($ORCHESTRATOR_URL/health)"
    echo "  → Check if service is running: docker-compose logs orchestrator"
fi

# Health check: Docling
if curl -sf "$DOCLING_URL/health" > /dev/null 2>&1; then
    pass "Docling service is healthy ($DOCLING_URL)"
else
    fail "Docling health check failed ($DOCLING_URL/health)"
    echo "  → Check if service is running: docker-compose logs docling"
fi

# Health check: Dashboard
if curl -sf "$DASHBOARD_URL" > /dev/null 2>&1; then
    pass "Dashboard is accessible ($DASHBOARD_URL)"
else
    warn "Dashboard health check failed ($DASHBOARD_URL)"
    echo "  → Check if service is running: docker-compose logs dashboard"
fi

# Health check: Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" PING 2>/dev/null | grep -q "PONG"; then
        pass "Redis is accessible"
    else
        fail "Redis connection failed"
        echo "  → Check Redis: docker-compose logs redis-stack"
    fi
else
    warn "redis-cli not found (skipping Redis check)"
fi

# ===========================================
# Section 3: Security Validation
# ===========================================
section "Security Validation"

# Check security headers
SECURITY_HEADERS=(
    "X-Content-Type-Options"
    "X-Frame-Options"
    "Strict-Transport-Security"
    "X-DNS-Prefetch-Control"
)

for HEADER in "${SECURITY_HEADERS[@]}"; do
    if curl -sI "$ORCHESTRATOR_URL/health" | grep -qi "$HEADER"; then
        pass "Security header present: $HEADER"
    else
        warn "Security header missing: $HEADER"
    fi
done

# Check if running over HTTPS in production
if [ "$ENVIRONMENT" = "production" ]; then
    if [[ "$ORCHESTRATOR_URL" =~ ^https:// ]]; then
        pass "Using HTTPS in production"
    else
        fail "NOT using HTTPS in production environment"
        echo "  → Production MUST use HTTPS"
    fi
fi

# Check for .git directory in deployment
if [ -d "$PROJECT_ROOT/.git" ] && [ "$ENVIRONMENT" = "production" ]; then
    warn ".git directory present in production deployment"
    echo "  → Consider removing .git for security"
fi

# ===========================================
# Section 4: Database & Persistence
# ===========================================
section "Database & Persistence"

# Check Redis indexes
if command -v redis-cli &> /dev/null; then
    INDEXES=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" FT._LIST 2>/dev/null || echo "")
    if [ -n "$INDEXES" ]; then
        INDEX_COUNT=$(echo "$INDEXES" | wc -l)
        pass "Redis indexes created (count: $INDEX_COUNT)"
    else
        warn "No Redis indexes found (may be created on first run)"
    fi
else
    warn "Cannot check Redis indexes (redis-cli not available)"
fi

# Check backup directory exists
BACKUP_DIR=${BACKUP_DIR:-/var/backups/mother-harness/redis}
if [ -d "$BACKUP_DIR" ]; then
    pass "Backup directory exists ($BACKUP_DIR)"

    # Check for recent backups
    RECENT_BACKUP=$(find "$BACKUP_DIR" -name "redis_backup_*.tar.gz" -mtime -1 2>/dev/null | head -1)
    if [ -n "$RECENT_BACKUP" ]; then
        pass "Recent backup found (< 24 hours old)"
    else
        warn "No backup found in last 24 hours"
        echo "  → Run backup: ./scripts/backup-redis.sh"
    fi
else
    warn "Backup directory not found ($BACKUP_DIR)"
    echo "  → Create: mkdir -p $BACKUP_DIR"
fi

# ===========================================
# Section 5: Dependencies & Build
# ===========================================
section "Dependencies & Build"

# Check if node_modules exists
if [ -d "$PROJECT_ROOT/node_modules" ]; then
    pass "Root dependencies installed"
else
    fail "Root dependencies not installed"
    echo "  → Run: pnpm install"
fi

# Check if services are built
SERVICES=("orchestrator" "docling" "agents" "shared" "dashboard")
for SERVICE in "${SERVICES[@]}"; do
    SERVICE_DIR="$PROJECT_ROOT/services/$SERVICE"
    if [ -d "$SERVICE_DIR" ]; then
        if [ -d "$SERVICE_DIR/dist" ] || [ -d "$SERVICE_DIR/.next" ]; then
            pass "$SERVICE is built"
        else
            warn "$SERVICE is not built"
            echo "  → Run: cd services/$SERVICE && npm run build"
        fi
    fi
done

# ===========================================
# Section 6: Documentation
# ===========================================
section "Documentation"

REQUIRED_DOCS=(
    "docs/launch-readiness.md"
    "docs/security-baseline.md"
    "docs/troubleshooting-runbook.md"
    "docs/oncall-procedures.md"
    "docs/user-onboarding-quickstart.md"
    "DEPLOYMENT_CHECKLIST.md"
)

for DOC in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$PROJECT_ROOT/$DOC" ]; then
        pass "Documentation exists: $DOC"
    else
        fail "Documentation missing: $DOC"
    fi
done

# ===========================================
# Section 7: Monitoring & Observability
# ===========================================
section "Monitoring & Observability"

# Check if metrics endpoint is accessible
if curl -sf "$ORCHESTRATOR_URL/api/metrics/summary" > /dev/null 2>&1; then
    pass "Metrics endpoint accessible"
else
    warn "Metrics endpoint not accessible (may require auth)"
fi

# Check API documentation
if curl -sf "$ORCHESTRATOR_URL/documentation" > /dev/null 2>&1; then
    pass "API documentation (Swagger UI) accessible"
else
    warn "Swagger UI not accessible"
fi

# ===========================================
# Results Summary
# ===========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

# Determine overall status
if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Deployment is ready.${NC}"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}⚠ Deployment has warnings but no critical failures.${NC}"
    echo -e "${YELLOW}Review warnings before proceeding to production.${NC}"
    exit 0
else
    echo -e "${RED}✗ Deployment validation failed.${NC}"
    echo -e "${RED}Fix failed checks before deploying.${NC}"
    exit 1
fi
